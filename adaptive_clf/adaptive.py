"""Adaptive estimator: state initialization, online update, observation builder."""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

from .configs import AcrobotParams, AdaptiveConfig, AdaptiveState, Array
from .acrobot import acrobot_affine_terms, acrobot_dynamics_true


def init_adaptive_state(p: AcrobotParams, adapt_cfg: AdaptiveConfig) -> AdaptiveState:
    a_hat0 = jnp.asarray(0.5 * (p.a_min + p.a_max), dtype=jnp.float32)
    info0 = jnp.asarray(adapt_cfg.info_init, dtype=jnp.float32)
    radius0 = jnp.asarray(max(0.5 * (p.a_max - p.a_min), adapt_cfg.radius_floor), dtype=jnp.float32)
    return AdaptiveState(a_hat=a_hat0, info=info0, radius=radius0)


def adaptive_update_simple(state: AdaptiveState,
                           x: Array,
                           u: Array,
                           a_true: Array,
                           dt: float,
                           p: AcrobotParams,
                           adapt_cfg: AdaptiveConfig) -> AdaptiveState:
    """Simple online estimator for the first experiment.

    This uses simulator-accessible accelerations. It is a practical placeholder,
    not the final theorem-grade certified set update.
    """
    f, g, y = acrobot_affine_terms(x, p)
    xdot_true = acrobot_dynamics_true(x, u, a_true, p)

    qdd_true = xdot_true[2:]
    y_q = y[2:]
    qdd_nom = f[2:] + g[2:] * u
    residual = qdd_true - (qdd_nom + y_q * state.a_hat)

    a_hat_next = state.a_hat + adapt_cfg.eta * jnp.dot(y_q, residual) * dt
    a_hat_next = jnp.clip(a_hat_next, p.a_min, p.a_max)

    info_next = state.info + jnp.dot(y_q, y_q) * dt
    radius_next = adapt_cfg.radius_scale / jnp.sqrt(info_next + 1e-6)
    radius_next = jnp.maximum(radius_next, adapt_cfg.radius_floor)

    return AdaptiveState(a_hat=a_hat_next, info=info_next, radius=radius_next)


def make_policy_observation(x: Array,
                            adaptive_state: AdaptiveState,
                            adapt_cfg: AdaptiveConfig) -> Array:
    a_hat_obs = jax.lax.stop_gradient(adaptive_state.a_hat) if adapt_cfg.stopgrad_obs else adaptive_state.a_hat
    rad_obs = jax.lax.stop_gradient(adaptive_state.radius) if adapt_cfg.stopgrad_obs else adaptive_state.radius
    return jnp.concatenate([x, jnp.asarray([a_hat_obs, rad_obs])], axis=0)
