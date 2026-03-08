"""Dubins car path-following dynamics with side-slip uncertainty.

Error-frame formulation for tracking a straight reference path at constant
speed v_ref along the x-axis.

State:   e = [e_x, e_y, e_theta]  (along-track, lateral, heading errors)
Control: u = [v, omega]            (forward speed, turn rate)
Uncertainty: a = side-slip velocity (scalar, unknown constant)

Dynamics:
    e_x_dot   = v cos(e_theta) - a sin(e_theta) - v_ref
    e_y_dot   = v sin(e_theta) + a cos(e_theta)
    e_theta_dot = omega

Control-affine form:  edot = f(e) + g(e) @ u + y(e) * a
    f(e) = [-v_ref, 0, 0]
    g(e) = [[cos(e_theta), 0], [sin(e_theta), 0], [0, 1]]
    y(e) = [-sin(e_theta), cos(e_theta), 0]

The side-slip is *unmatched*: it enters e_y but the control omega only
enters e_theta.  Compensating slip requires turning then driving ---
exactly the underactuation structure we want for the CLF shield.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

try:
    import scipy.linalg as spla
except Exception:
    spla = None

from .configs import Array, DubinsParams


def dubins_affine_terms(x: Array, p: DubinsParams) -> Tuple[Array, Array, Array]:
    """Return f(x), g(x), y(x) for the Dubins path-following error dynamics.

    f: (3,), g: (3, 2), y: (3,)
    """
    _ex, _ey, eth = x
    ct = jnp.cos(eth)
    st = jnp.sin(eth)

    f = jnp.array([-p.v_ref, 0.0, 0.0])

    g = jnp.array([
        [ct, 0.0],
        [st, 0.0],
        [0.0, 1.0],
    ])

    y = jnp.array([-st, ct, 0.0])

    return f, g, y


def dubins_dynamics(x: Array, u: Array, a_true: Array,
                    p: DubinsParams) -> Array:
    """Full nonlinear error dynamics: edot = f + g @ u + y * a."""
    f, g, y = dubins_affine_terms(x, p)
    return f + g @ u + y * a_true


def rk4_step_dubins(x: Array, u: Array, a_true: Array,
                    dt: float, p: DubinsParams) -> Array:
    k1 = dubins_dynamics(x, u, a_true, p)
    k2 = dubins_dynamics(x + 0.5 * dt * k1, u, a_true, p)
    k3 = dubins_dynamics(x + 0.5 * dt * k2, u, a_true, p)
    k4 = dubins_dynamics(x + dt * k3, u, a_true, p)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def linearize_dubins(p: DubinsParams, a_nom: float = 0.0) -> Tuple[Array, Array]:
    """Linearize error dynamics at e=0, u=[v_ref, 0].

    Returns (A, B) where A is (3,3) and B is (3,2).
    """
    x_eq = jnp.zeros(3)
    u_eq = jnp.array([p.v_ref, 0.0])
    a_nom_arr = jnp.asarray(a_nom, dtype=jnp.float32)

    A = jax.jacfwd(lambda x_: dubins_dynamics(x_, u_eq, a_nom_arr, p))(x_eq)
    B = jax.jacfwd(lambda u_: dubins_dynamics(x_eq, u_, a_nom_arr, p))(u_eq)
    return A, B


def solve_dubins_lqr(p: DubinsParams,
                     Q: Array,
                     R: Array,
                     a_nom: float = 0.0) -> Tuple[Array, Array]:
    """Return (P, K) from the nominal linearization and CARE.

    K is the LQR gain such that delta_u = -K @ e, and
    u_lqr = [v_ref, 0] - K @ e.
    """
    if spla is None:
        raise ImportError("scipy required for solve_dubins_lqr")

    A, B = linearize_dubins(p, a_nom=a_nom)
    P = spla.solve_continuous_are(
        jnp.asarray(A), jnp.asarray(B), jnp.asarray(Q), jnp.asarray(R),
    )
    K = jnp.linalg.solve(jnp.asarray(R), jnp.asarray(B).T @ P)
    return jnp.asarray(P, dtype=jnp.float32), jnp.asarray(K, dtype=jnp.float32)
