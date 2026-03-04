"""Acrobot dynamics in control-affine form, linearization, and LQR."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

try:
    import scipy.linalg as spla
except Exception:  # pragma: no cover - optional dependency
    spla = None

from .configs import AcrobotParams, Array


def acrobot_terms(x: Array, p: AcrobotParams) -> Tuple[Array, Array, Array]:
    """Manipulator matrices M(q), C(q,qdot), tau_g(q) in local upright coordinates.

    Follows Tedrake (Underactuated Robotics, Ch.3): I1, I2 are moments of
    inertia about the respective pivot joints, not the centers of mass.

    x = [delta_q1, delta_q2, q1dot, q2dot]
    q1 = pi + delta_q1
    q2 = delta_q2
    """
    dq1, dq2, w1, w2 = x
    q1 = jnp.pi + dq1
    q2 = dq2

    c2 = jnp.cos(q2)
    s2 = jnp.sin(q2)
    s1 = jnp.sin(q1)
    s12 = jnp.sin(q1 + q2)

    h = p.m2 * p.l1 * p.lc2 * c2
    M11 = p.I1 + p.I2 + p.m2 * p.l1 ** 2 + 2.0 * h
    M12 = p.I2 + h
    M22 = p.I2
    M = jnp.array([[M11, M12], [M12, M22]])

    C = jnp.array([
        [-2.0 * p.m2 * p.l1 * p.lc2 * s2 * w2, -p.m2 * p.l1 * p.lc2 * s2 * w2],
        [p.m2 * p.l1 * p.lc2 * s2 * w1, 0.0],
    ])

    tau_g = jnp.array([
        -p.m1 * p.g * p.lc1 * s1 - p.m2 * p.g * (p.l1 * s1 + p.lc2 * s12),
        -p.m2 * p.g * p.lc2 * s12,
    ])
    return M, C, tau_g


def acrobot_affine_terms(x: Array, p: AcrobotParams) -> Tuple[Array, Array, Array]:
    """Return f(x), g(x), y(x) for the uncertain Acrobot.

    Shoulder damping uncertainty:
        tau_f = [a_true * q1dot, 0]^T
    so that
        xdot = f(x) + g(x) u + y(x) a_true.
    """
    _, _, w1, w2 = x
    qdot = jnp.array([w1, w2])

    M, C, tau_g = acrobot_terms(x, p)
    reg = max(float(p.eps), 1e-6)
    M_reg = M + reg * jnp.eye(2, dtype=M.dtype)

    rhs = jnp.column_stack([
        tau_g - C @ qdot,
        jnp.array([0.0, 1.0]),
        jnp.array([-w1, 0.0]),
    ])
    qdd = jnp.linalg.solve(M_reg, rhs)
    model_qdd, input_qdd, friction_qdd = qdd[:, 0], qdd[:, 1], qdd[:, 2]

    f = jnp.concatenate([qdot, model_qdd])
    g = jnp.concatenate([jnp.zeros(2), input_qdd])
    y = jnp.concatenate([jnp.zeros(2), friction_qdd])
    return f, g, y


def acrobot_dynamics_true(x: Array, u: Array, a_true: Array, p: AcrobotParams) -> Array:
    f, g, y = acrobot_affine_terms(x, p)
    return f + g * u + y * a_true


def rk4_step(dynamics_fn: Callable[[Array, Array, Array, AcrobotParams], Array],
             x: Array, u: Array, a_true: Array, dt: float, p: AcrobotParams) -> Array:
    k1 = dynamics_fn(x, u, a_true, p)
    k2 = dynamics_fn(x + 0.5 * dt * k1, u, a_true, p)
    k3 = dynamics_fn(x + 0.5 * dt * k2, u, a_true, p)
    k4 = dynamics_fn(x + dt * k3, u, a_true, p)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def linearize_acrobot_at_upright(p: AcrobotParams, a_nom: float = 0.0) -> Tuple[Array, Array]:
    """Linearize xdot = f(x) + g(x) u + y(x) a_nom at x=0, u=0."""
    x_eq = jnp.zeros((4,))
    u_eq = jnp.array(0.0)

    def fx(x_: Array) -> Array:
        return acrobot_dynamics_true(x_, u_eq, a_nom, p)

    def fu(u_: Array) -> Array:
        return acrobot_dynamics_true(x_eq, u_, a_nom, p)

    A = jax.jacfwd(fx)(x_eq)
    B = jax.jacfwd(fu)(u_eq).reshape(4, 1)
    return A, B


def solve_lqr_P(p: AcrobotParams,
                Q: Array,
                R: Array,
                a_nom: float = 0.0) -> Tuple[Array, Array]:
    """Return (P, K) from the nominal linearization and CARE.

    Requires SciPy for solve_continuous_are.
    """
    if spla is None:  # pragma: no cover
        raise ImportError("scipy is required for solve_lqr_P but is not installed.")

    A, B = linearize_acrobot_at_upright(p, a_nom=a_nom)
    A_np = jnp.asarray(A)
    B_np = jnp.asarray(B)
    Q_np = jnp.asarray(Q)
    R_np = jnp.asarray(R)

    P = spla.solve_continuous_are(A_np, B_np, Q_np, R_np)
    K = jnp.linalg.solve(R_np, B_np.T @ P)
    return jnp.asarray(P), jnp.asarray(K)
