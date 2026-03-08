"""Cart-pole dynamics in control-affine form, linearization, and LQR.

State convention: x = [x_cart, theta, x_cart_dot, theta_dot]
  - theta = 0 is upright (goal), theta = pi is hanging down.
  - Force u is applied horizontally to the cart.

Uncertainty model: cart friction  tau_f = [a_true * x_cart_dot, 0]^T.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

try:
    import scipy.linalg as spla
except Exception:  # pragma: no cover
    spla = None

from .configs import Array, CartPoleParams


def cartpole_affine_terms(x: Array, p: CartPoleParams) -> Tuple[Array, Array, Array]:
    """Return f(x), g(x), y(x) for the cart-pole with cart friction uncertainty.

    xdot = f(x) + g(x) u + y(x) a_true
    """
    x_c, theta, xdot, thetadot = x
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    M = jnp.array([
        [p.mc + p.mp,      p.mp * p.l * ct],
        [p.mp * p.l * ct,  p.mp * p.l ** 2],
    ])

    model_rhs = jnp.array([
        p.mp * p.l * st * thetadot ** 2,
        p.mp * p.g * p.l * st,
    ])
    input_rhs = jnp.array([1.0, 0.0])
    friction_rhs = jnp.array([-xdot, 0.0])

    reg = max(float(p.eps), 1e-8)
    M_reg = M + reg * jnp.eye(2, dtype=M.dtype)

    rhs = jnp.column_stack([model_rhs, input_rhs, friction_rhs])
    qdd = jnp.linalg.solve(M_reg, rhs)
    model_qdd, input_qdd, friction_qdd = qdd[:, 0], qdd[:, 1], qdd[:, 2]

    qdot = jnp.array([xdot, thetadot])
    f = jnp.concatenate([qdot, model_qdd])
    g = jnp.concatenate([jnp.zeros(2), input_qdd])
    y = jnp.concatenate([jnp.zeros(2), friction_qdd])
    return f, g, y


def cartpole_dynamics(x: Array, u: Array, a_true: Array,
                      p: CartPoleParams) -> Array:
    f, g, y = cartpole_affine_terms(x, p)
    return f + g * u + y * a_true


def rk4_step_cartpole(x: Array, u: Array, a_true: Array,
                       dt: float, p: CartPoleParams) -> Array:
    k1 = cartpole_dynamics(x, u, a_true, p)
    k2 = cartpole_dynamics(x + 0.5 * dt * k1, u, a_true, p)
    k3 = cartpole_dynamics(x + 0.5 * dt * k2, u, a_true, p)
    k4 = cartpole_dynamics(x + dt * k3, u, a_true, p)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def linearize_cartpole_at_upright(p: CartPoleParams,
                                   a_nom: float = 0.0) -> Tuple[Array, Array]:
    """Linearize xdot = f(x) + g(x)*u + y(x)*a_nom at x=0, u=0."""
    x_eq = jnp.zeros((4,))
    u_eq = jnp.array(0.0)

    def fx(x_: Array) -> Array:
        return cartpole_dynamics(x_, u_eq, a_nom, p)

    def fu(u_: Array) -> Array:
        return cartpole_dynamics(x_eq, u_, a_nom, p)

    A = jax.jacfwd(fx)(x_eq)
    B = jax.jacfwd(fu)(u_eq).reshape(4, 1)
    return A, B


def solve_cartpole_lqr(p: CartPoleParams,
                        Q: Array,
                        R: Array,
                        a_nom: float = 0.0) -> Tuple[Array, Array]:
    """Return (P, K) from the nominal linearization and CARE."""
    if spla is None:  # pragma: no cover
        raise ImportError("scipy required for solve_cartpole_lqr")

    A, B = linearize_cartpole_at_upright(p, a_nom=a_nom)
    P = spla.solve_continuous_are(
        jnp.asarray(A), jnp.asarray(B), jnp.asarray(Q), jnp.asarray(R),
    )
    K = jnp.linalg.solve(jnp.asarray(R), jnp.asarray(B).T @ P)
    return jnp.asarray(P), jnp.asarray(K)
