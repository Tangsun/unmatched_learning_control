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
                        x_hat=None, w=None, eta=None) -> AdaptiveState:
    """Build an AdaptiveState, filling observer fields with zeros if omitted."""
    n = max(state_dim, 1)
    zeros = jnp.zeros(n, dtype=jnp.float32)
    return AdaptiveState(
        a_hat=jnp.asarray(a_hat, dtype=jnp.float32),
        info=jnp.asarray(info, dtype=jnp.float32),
        radius=jnp.asarray(radius, dtype=jnp.float32),
        x_hat=zeros if x_hat is None else jnp.asarray(x_hat, dtype=jnp.float32),
        w=zeros if w is None else jnp.asarray(w, dtype=jnp.float32),
        eta=zeros if eta is None else jnp.asarray(eta, dtype=jnp.float32),
    )


def init_adaptive_state(p: Any, adapt_cfg: AdaptiveConfig,
                        state_dim: int = 3,
                        x0: Optional[Array] = None) -> AdaptiveState:
    """Initialize adaptive state.

    For observer mode, x0 is the initial measured state (used for x_hat(0)=x0
    so that e(0)=0 and eta(0)=0).
    """
    if not adapt_cfg.adapt_enabled:
        return make_adaptive_state(0.0, 1e-6, 0.0, state_dim=state_dim)

    a_hat0 = 0.5 * (p.a_min + p.a_max)
    radius0 = max(0.5 * (p.a_max - p.a_min), adapt_cfg.radius_floor)

    if adapt_cfg.use_observer:
        # Observer: x_hat(0)=x0 so e(0)=0, w(0)=0, eta(0)=e(0)=0
        x_hat0 = jnp.zeros(state_dim) if x0 is None else x0
        w0 = jnp.zeros(state_dim, dtype=jnp.float32)
        eta0 = jnp.zeros(state_dim, dtype=jnp.float32)
        return make_adaptive_state(
            a_hat0, adapt_cfg.info_init, radius0,
            state_dim=state_dim,
            x_hat=x_hat0, w=w0, eta=eta0,
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
                             ) -> AdaptiveState:
    """Observer-based adaptive update using only state measurements.

    Implements the scheme from Section 2.1:
      Predictor: x_hat_dot = f(x) + g(x) u + Y(x) a_hat + k * e
      Filter:    w_dot = Y(x) - k * w
      Auxiliary:  eta_dot = -k * eta
      Update:    a_hat_dot = gamma * w^T * (e - eta)

    Uses exponential integrator for the linear parts (w, eta) to avoid
    stiffness with large k. The predictor uses Euler.

    Does NOT require a_true — only uses measured state x.
    """
    if affine_terms_fn is None:
        raise ValueError("affine_terms_fn is required")

    k = adapt_cfg.observer_k
    gamma = adapt_cfg.observer_gamma
    eps_w = adapt_cfg.observer_eps_w
    margin = adapt_cfg.observer_radius_margin

    f, g, y = affine_terms_fn(x, p)

    # Prediction error (wrap angle if needed — handled by caller wrapping x)
    e = x - state.x_hat

    # --- Exponential integrator for eta: eta_dot = -k * eta ---
    # Exact: eta(t+dt) = eta(t) * exp(-k*dt)
    decay = jnp.exp(-k * dt)
    eta_next = state.eta * decay

    # --- Exponential integrator for w: w_dot = Y(x) - k * w ---
    # Exact for constant Y over [t, t+dt]:
    #   w(t+dt) = w(t)*exp(-k*dt) + Y(x)/k * (1 - exp(-k*dt))
    w_next = state.w * decay + y * (1.0 - decay) / k

    # --- Predictor: x_hat_dot = f(x) + g(x) u + Y(x) a_hat + k * e ---
    # Euler step (coupling to a_hat makes exponential integrator impractical)
    x_hat_dot = f + g @ u + y * state.a_hat + k * e
    x_hat_next = state.x_hat + x_hat_dot * dt

    # --- Parameter update: a_hat_dot = gamma * w^T * (e - eta) ---
    innovation = e - state.eta
    a_hat_dot = gamma * jnp.dot(state.w, innovation)
    a_hat_next = state.a_hat + a_hat_dot * dt
    a_hat_next = jnp.clip(a_hat_next, p.a_min, p.a_max)

    # --- Radius from observer: |a_tilde_est| = |w^T(e - eta)| / max(||w||^2, eps_w) ---
    # Only trust the estimate when ||w||^2 > eps_w (filter has accumulated signal).
    # When w is too small, keep the prior radius to avoid false confidence.
    w_norm_sq = jnp.dot(state.w, state.w)
    a_tilde_est = jnp.dot(state.w, innovation) / jnp.maximum(w_norm_sq, eps_w)
    r_raw = jnp.abs(a_tilde_est) + margin
    w_ready = w_norm_sq > eps_w
    r_candidate = jnp.where(w_ready, r_raw, state.radius)

    # Monotonic envelope: radius can only shrink
    radius_next = jnp.minimum(state.radius, r_candidate)
    radius_next = jnp.maximum(radius_next, adapt_cfg.radius_floor)

    # Keep info updated for diagnostics (accumulated ||Y||^2)
    info_next = state.info + jnp.dot(y, y) * dt

    return AdaptiveState(
        a_hat=a_hat_next,
        info=info_next,
        radius=radius_next,
        x_hat=x_hat_next,
        w=w_next,
        eta=eta_next,
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
