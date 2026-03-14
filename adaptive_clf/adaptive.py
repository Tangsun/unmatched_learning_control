"""Adaptive estimator: state initialization, online update, observation builder.

Includes both the legacy heuristic estimator (adaptive_update_generic) and
the observer-based scheme from Section 2.1 of main.pdf (adaptive_update_observer).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from .configs import AdaptiveConfig, AdaptiveState, Array


# ---------------------------------------------------------------------------
# Helper: construct AdaptiveState with correct shapes
# ---------------------------------------------------------------------------

def make_adaptive_state(a_hat, info, radius, state_dim: int = 0,
                        x_hat=None, w=None, eta=None,
                        a_hat_internal=None, info_internal=None,
                        q_internal=None, ve0=None) -> AdaptiveState:
    """Build an AdaptiveState, filling observer/internal fields if omitted."""
    n = max(state_dim, 1)
    zeros = jnp.zeros(n, dtype=jnp.float32)
    a_hat_arr = jnp.asarray(a_hat, dtype=jnp.float32)
    info_arr = jnp.asarray(info, dtype=jnp.float32)
    return AdaptiveState(
        a_hat=a_hat_arr,
        info=info_arr,
        radius=jnp.asarray(radius, dtype=jnp.float32),
        x_hat=zeros if x_hat is None else jnp.asarray(x_hat, dtype=jnp.float32),
        w=zeros if w is None else jnp.asarray(w, dtype=jnp.float32),
        eta=zeros if eta is None else jnp.asarray(eta, dtype=jnp.float32),
        a_hat_internal=a_hat_arr if a_hat_internal is None else jnp.asarray(a_hat_internal, dtype=jnp.float32),
        info_internal=info_arr if info_internal is None else jnp.asarray(info_internal, dtype=jnp.float32),
        q_internal=jnp.array(0.0, dtype=jnp.float32) if q_internal is None else jnp.asarray(q_internal, dtype=jnp.float32),
        ve0=info_arr if ve0 is None else jnp.asarray(ve0, dtype=jnp.float32),
    )


def init_adaptive_state(p: Any, adapt_cfg: AdaptiveConfig,
                        state_dim: int = 3,
                        x0: Optional[Array] = None,
                        a_range: Optional[float] = None) -> AdaptiveState:
    """Initialize adaptive state.

    For observer mode, x0 is the initial measured state (used for x_hat(0)=x0
    so that e(0)=0 and eta(0)=0).

    a_range: if provided, use this as the half-width of the uncertainty set
             instead of the system-wide (p.a_min, p.a_max) bounds.
    """
    if not adapt_cfg.adapt_enabled:
        return make_adaptive_state(0.0, 1e-6, 0.0, state_dim=state_dim)

    if a_range is not None:
        a_hat0 = 0.0
        radius0 = a_range if adapt_cfg.use_observer else max(a_range, adapt_cfg.radius_floor)
    else:
        a_hat0 = 0.5 * (p.a_min + p.a_max)
        base_radius0 = 0.5 * (p.a_max - p.a_min)
        radius0 = base_radius0 if adapt_cfg.use_observer else max(base_radius0, adapt_cfg.radius_floor)

    if adapt_cfg.use_observer:
        # Observer: x_hat(0)=x0 so e(0)=0, w(0)=0, eta(0)=e(0)=0
        x_hat0 = jnp.zeros(state_dim) if x0 is None else x0
        w0 = jnp.zeros(state_dim, dtype=jnp.float32)
        eta0 = jnp.zeros(state_dim, dtype=jnp.float32)
        # info/info_internal store the published/internal V_eη values.
        # V_eη(0) = ½ z_0² so z_θ^eη(0) = √V_eη(0) = z_0/√2.
        # This is intentionally < radius_0; the gap provides headroom
        # for the joint acceptance condition (Algorithm 1).
        V_eη_0 = 0.5 * radius0 ** 2
        return make_adaptive_state(
            a_hat0, V_eη_0, radius0,
            state_dim=state_dim,
            x_hat=x_hat0, w=w0, eta=eta0,
            a_hat_internal=a_hat0, info_internal=V_eη_0,
            q_internal=0.0, ve0=V_eη_0,
        )
    else:
        info0 = adapt_cfg.info_init
        return make_adaptive_state(
            a_hat0, info0, radius0, state_dim=state_dim,
        )


# ---------------------------------------------------------------------------
# Legacy heuristic estimators (use xdot_true — not available in practice)
# ---------------------------------------------------------------------------

def adaptive_update_simple(state: AdaptiveState,
                           x: Array,
                           u: Array,
                           a_true: Array,
                           dt: float,
                           p: Any,
                           adapt_cfg: AdaptiveConfig,
                           affine_terms_fn: Optional[Callable] = None,
                           dynamics_fn: Optional[Callable] = None,
                           ) -> AdaptiveState:
    """Simple online estimator for the first experiment.

    This uses simulator-accessible accelerations. It is a practical placeholder,
    not the final theorem-grade certified set update.
    """
    if affine_terms_fn is None:
        raise ValueError("affine_terms_fn is required")
    if dynamics_fn is None:
        raise ValueError("dynamics_fn is required")
    f, g, y = affine_terms_fn(x, p)
    xdot_true = dynamics_fn(x, u, a_true, p)

    qdd_true = xdot_true[2:]
    y_q = y[2:]
    qdd_nom = f[2:] + g[2:] * u
    residual = qdd_true - (qdd_nom + y_q * state.a_hat)

    a_hat_next = state.a_hat + adapt_cfg.eta * jnp.dot(y_q, residual) * dt
    a_hat_next = jnp.clip(a_hat_next, p.a_min, p.a_max)

    info_next = state.info + jnp.dot(y_q, y_q) * dt
    radius_next = adapt_cfg.radius_scale / jnp.sqrt(info_next + 1e-6)
    radius_next = jnp.maximum(radius_next, adapt_cfg.radius_floor)

    return state._replace(a_hat=a_hat_next, info=info_next, radius=radius_next)


def adaptive_update_generic(state: AdaptiveState,
                            x: Array,
                            u: Array,
                            a_true: Array,
                            dt: float,
                            p: Any,
                            adapt_cfg: AdaptiveConfig,
                            affine_terms_fn: Optional[Callable] = None,
                            dynamics_fn: Optional[Callable] = None,
                            ) -> AdaptiveState:
    """Generic adaptive update using full-state residual.

    Works for any control-affine system: xdot = f(x) + g(x) @ u + y(x) * a.
    Update law: a_hat += eta * y^T * (xdot_true - xdot_model) * dt.
    """
    if affine_terms_fn is None:
        raise ValueError("affine_terms_fn is required")
    if dynamics_fn is None:
        raise ValueError("dynamics_fn is required")
    f, g, y = affine_terms_fn(x, p)
    xdot_true = dynamics_fn(x, u, a_true, p)

    xdot_model = f + g @ u + y * state.a_hat
    residual = xdot_true - xdot_model  # = y * (a_true - a_hat)

    a_hat_next = state.a_hat + adapt_cfg.eta * jnp.dot(y, residual) * dt
    a_hat_next = jnp.clip(a_hat_next, p.a_min, p.a_max)

    info_next = state.info + jnp.dot(y, y) * dt
    radius_next = adapt_cfg.radius_scale / jnp.sqrt(info_next + 1e-6)
    radius_next = jnp.maximum(radius_next, adapt_cfg.radius_floor)

    return state._replace(a_hat=a_hat_next, info=info_next, radius=radius_next)


# ---------------------------------------------------------------------------
# Observer-based adaptation (Section 2.1 of main.pdf)
# ---------------------------------------------------------------------------

def adaptive_update_observer(state: AdaptiveState,
                             x: Array,
                             u: Array,
                             dt: float,
                             p: Any,
                             adapt_cfg: AdaptiveConfig,
                             affine_terms_fn: Optional[Callable] = None,
                             dynamics_fn: Optional[Callable] = None,
                             a_true: Optional[Array] = None,
                             ) -> AdaptiveState:
    """Observer-based adaptive update using only state measurements.

    Implements the scheme from Section 2.1:
      Predictor: x_hat_dot = f(x) + g(x) u + Y(x) a_hat + w * a_hat_dot + k * e
      Filter:    w_dot = Y(x) - k * w
      Auxiliary:  eta_dot = -k * eta
      Update:    a_hat_dot = gamma * w^T * (e - eta)

    Uses RK4 on the full continuous-time observer ODE. When dynamics_fn and
    a_true are provided, the observer is integrated against the simulated
    plant trajectory x(t) over the sample interval, matching the paper's
    continuous-time estimator more closely. Otherwise x is frozen over dt.

    Does NOT require a_true — only uses measured state x.
    """
    if affine_terms_fn is None:
        raise ValueError("affine_terms_fn is required")

    k = adapt_cfg.observer_k
    gamma = adapt_cfg.observer_gamma

    def observer_rhs(x_i, x_hat_i, w_i, eta_i, a_hat_i):
        f_i, g_i, y_i = affine_terms_fn(x_i, p)
        e_i = x_i - x_hat_i
        innovation_i = e_i - eta_i
        a_hat_dot_i = gamma * jnp.dot(w_i, innovation_i)
        x_hat_dot_i = f_i + g_i @ u + y_i * a_hat_i + w_i * a_hat_dot_i + k * e_i
        w_dot_i = y_i - k * w_i
        eta_dot_i = -k * eta_i
        info_dot_i = -gamma * jnp.dot(innovation_i, innovation_i)
        q_dot_i = jnp.dot(w_i, w_i)
        if dynamics_fn is not None and a_true is not None:
            x_dot_i = dynamics_fn(x_i, u, a_true, p)
        else:
            x_dot_i = jnp.zeros_like(x_i)
        return x_dot_i, x_hat_dot_i, w_dot_i, eta_dot_i, a_hat_dot_i, info_dot_i, q_dot_i

    k1_x, k1_xhat, k1_w, k1_eta, k1_a, k1_info, k1_q = observer_rhs(
        x, state.x_hat, state.w, state.eta, state.a_hat_internal,
    )
    a2 = state.a_hat_internal + 0.5 * dt * k1_a
    x2 = x + 0.5 * dt * k1_x
    k2_x, k2_xhat, k2_w, k2_eta, k2_a, k2_info, k2_q = observer_rhs(
        x2,
        state.x_hat + 0.5 * dt * k1_xhat,
        state.w + 0.5 * dt * k1_w,
        state.eta + 0.5 * dt * k1_eta,
        a2,
    )
    a3 = state.a_hat_internal + 0.5 * dt * k2_a
    x3 = x + 0.5 * dt * k2_x
    k3_x, k3_xhat, k3_w, k3_eta, k3_a, k3_info, k3_q = observer_rhs(
        x3,
        state.x_hat + 0.5 * dt * k2_xhat,
        state.w + 0.5 * dt * k2_w,
        state.eta + 0.5 * dt * k2_eta,
        a3,
    )
    a4 = state.a_hat_internal + dt * k3_a
    x4 = x + dt * k3_x
    k4_x, k4_xhat, k4_w, k4_eta, k4_a, k4_info, k4_q = observer_rhs(
        x4,
        state.x_hat + dt * k3_xhat,
        state.w + dt * k3_w,
        state.eta + dt * k3_eta,
        a4,
    )

    x_hat_next = state.x_hat + (dt / 6.0) * (
        k1_xhat + 2.0 * k2_xhat + 2.0 * k3_xhat + k4_xhat
    )
    w_next = state.w + (dt / 6.0) * (
        k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w
    )
    eta_next = state.eta + (dt / 6.0) * (
        k1_eta + 2.0 * k2_eta + 2.0 * k3_eta + k4_eta
    )
    a_hat_internal_next = state.a_hat_internal + (dt / 6.0) * (
        k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a
    )

    # --- Lyapunov-based radius bound (Adetola et al. 2009, eqs 14-16) ---
    # state.info_internal stores the continuously evolving V_eη (eq 15b),
    # while state.q_internal stores Q from eq. (8). The paper radius is
    # zθ = min(zθ^eη, zθ^E) from eq. (14).
    info_internal_next = state.info_internal + (dt / 6.0) * (
        k1_info + 2.0 * k2_info + 2.0 * k3_info + k4_info
    )
    info_internal_next = jnp.maximum(info_internal_next, 0.0)
    q_internal_next = state.q_internal + (dt / 6.0) * (
        k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q
    )

    z_eη = jnp.sqrt(info_internal_next)  # eq. (15a)
    alpha = 1.0 / (1.0 + gamma * q_internal_next)  # eq. (9), scalar case
    V_E = alpha * state.ve0                               # eq. (16b)
    z_E = jnp.sqrt(V_E)                                  # eq. (16a)
    r_candidate = jnp.minimum(z_eη, z_E)                 # eq. (14)

    # --- Joint acceptance (Algorithm 1 from Adetola et al. 2009) ---
    # The continuous observer state keeps evolving, but the controller-facing
    # (a_hat, radius) pair is only published when the new ball is contained in
    # the previous published ball: r_new <= r_old - |a_hat_new - a_hat_old|.
    delta_a = jnp.abs(a_hat_internal_next - state.a_hat)
    accept = r_candidate <= state.radius - delta_a
    a_hat_next = jnp.where(accept, a_hat_internal_next, state.a_hat)
    radius_next = jnp.where(accept, r_candidate, state.radius)
    info_next = jnp.where(accept, info_internal_next, state.info)

    return AdaptiveState(
        a_hat=a_hat_next,
        info=info_next,
        radius=radius_next,
        x_hat=x_hat_next,
        w=w_next,
        eta=eta_next,
        a_hat_internal=a_hat_internal_next,
        info_internal=info_internal_next,
        q_internal=q_internal_next,
        ve0=state.ve0,
    )


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def make_policy_observation(x: Array,
                            adaptive_state: AdaptiveState,
                            adapt_cfg: AdaptiveConfig) -> Array:
    if not adapt_cfg.adapt_enabled:
        return x
    a_hat_obs = jax.lax.stop_gradient(adaptive_state.a_hat) if adapt_cfg.stopgrad_obs else adaptive_state.a_hat
    rad_obs = jax.lax.stop_gradient(adaptive_state.radius) if adapt_cfg.stopgrad_obs else adaptive_state.radius
    return jnp.concatenate([x, jnp.asarray([a_hat_obs, rad_obs])], axis=0)
