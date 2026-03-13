"""Planar Vertical Take-Off and Landing (PVTOL) dynamics.

State:   x = [px, py, theta, vx, vy, thetadot]
  - px, py: position in inertial frame
  - theta: roll angle (0 = upright)
  - vx, vy: velocity in inertial frame
  - thetadot: angular rate

Control: u = [T, tau]
  - T: total thrust (along body z-axis)
  - tau: torque about body center

Uncertainty: a = lateral wind acceleration (scalar, unknown constant)

Dynamics:
    px_dot     = vx
    py_dot     = vy
    theta_dot  = thetadot
    vx_dot     = -(T/m) sin(theta) + a
    vy_dot     =  (T/m) cos(theta) - g
    thetadot_dot = tau / J

Control-affine form:  xdot = f(x) + G(x) @ u + Y(x) * a
    f(x) = [vx, vy, thetadot, 0, -g, 0]
    G(x) = [[0, 0], [0, 0], [0, 0],
            [-sin(theta)/m, 0], [cos(theta)/m, 0], [0, 1/J]]
    Y(x) = [0, 0, 0, 1, 0, 0]

The wind is *unmatched*: it directly affects vx, but thrust acts along the
body axis (-sin(theta), cos(theta))/m.  To compensate lateral wind the
vehicle must tilt into the wind and increase thrust --- exactly the
underactuation structure motivating adaptive CLF.

Hover equilibrium: x=0, u=[m*g, 0], theta=0.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

try:
    import scipy.linalg as spla
except Exception:
    spla = None

from .configs import Array, PVTOLParams


def pvtol_affine_terms(x: Array, p: PVTOLParams) -> Tuple[Array, Array, Array]:
    """Return f(x), G(x), Y(x) for the PVTOL.

    f: (6,), G: (6, 2), Y: (6,)
    """
    _px, _py, theta, vx, vy, thetadot = x
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    f = jnp.array([vx, vy, thetadot, 0.0, -p.g, 0.0])

    G = jnp.array([
        [0.0,      0.0],
        [0.0,      0.0],
        [0.0,      0.0],
        [-st / p.m, 0.0],
        [ct / p.m,  0.0],
        [0.0,      1.0 / p.J],
    ])

    Y = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    return f, G, Y


def pvtol_dynamics(x: Array, u: Array, a_true: Array,
                   p: PVTOLParams) -> Array:
    """Full nonlinear dynamics: xdot = f + G @ u + Y * a."""
    f, G, Y = pvtol_affine_terms(x, p)
    return f + G @ u + Y * a_true


def rk4_step_pvtol(x: Array, u: Array, a_true: Array,
                   dt: float, p: PVTOLParams) -> Array:
    k1 = pvtol_dynamics(x, u, a_true, p)
    k2 = pvtol_dynamics(x + 0.5 * dt * k1, u, a_true, p)
    k3 = pvtol_dynamics(x + 0.5 * dt * k2, u, a_true, p)
    k4 = pvtol_dynamics(x + dt * k3, u, a_true, p)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def linearize_pvtol(p: PVTOLParams, a_nom: float = 0.0) -> Tuple[Array, Array]:
    """Linearize at hover: x=0, u=[m*g, 0].

    Returns (A, B) where A is (6,6) and B is (6,2).
    """
    x_eq = jnp.zeros(6)
    u_eq = jnp.array([p.m * p.g, 0.0])
    a_nom_arr = jnp.asarray(a_nom, dtype=jnp.float32)

    A = jax.jacfwd(lambda x_: pvtol_dynamics(x_, u_eq, a_nom_arr, p))(x_eq)
    B = jax.jacfwd(lambda u_: pvtol_dynamics(x_eq, u_, a_nom_arr, p))(u_eq)
    return A, B


def solve_pvtol_lqr(p: PVTOLParams,
                    Q: Array,
                    R: Array,
                    a_nom: float = 0.0) -> Tuple[Array, Array]:
    """Return (P, K) from the nominal linearization and CARE.

    K is the LQR gain such that delta_u = -K @ x, and
    u_lqr = [m*g, 0] - K @ x.
    """
    if spla is None:
        raise ImportError("scipy required for solve_pvtol_lqr")

    A, B = linearize_pvtol(p, a_nom=a_nom)
    P = spla.solve_continuous_are(
        jnp.asarray(A), jnp.asarray(B), jnp.asarray(Q), jnp.asarray(R),
    )
    K = jnp.linalg.solve(jnp.asarray(R), jnp.asarray(B).T @ P)
    return jnp.asarray(P, dtype=jnp.float32), jnp.asarray(K, dtype=jnp.float32)
