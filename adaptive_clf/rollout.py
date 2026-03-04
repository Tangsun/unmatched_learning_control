"""Training objective: stage cost, episode rollout, batched loss, and sampling."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from .configs import (
    AcrobotParams,
    AdaptiveConfig,
    Array,
    CLFConfig,
    LyapunovConfig,
    RolloutConfig,
)
from .nn import init_policy_params, policy_apply
from .acrobot import acrobot_dynamics_true, rk4_step
from .lyapunov import init_lyapunov_params
from .shield import clf_shield
from .adaptive import init_adaptive_state, adaptive_update_simple, make_policy_observation


def stage_cost(x: Array,
               u: Array,
               u_nom: Array,
               feasible: Array,
               rollout_cfg: RolloutConfig) -> Array:
    Q = jnp.asarray(rollout_cfg.Q_track)
    x_cost = x @ Q @ x
    return (
        x_cost
        + rollout_cfg.R_u * (u ** 2)
        + rollout_cfg.R_proj * ((u - u_nom) ** 2)
        + rollout_cfg.R_infeasible * (1.0 - feasible)
    )


def episode_rollout(policy_params: Dict[str, Any],
                    lyap_params: Dict[str, Any],
                    x0: Array,
                    a_true: Array,
                    p: AcrobotParams,
                    policy_hidden_sizes: Tuple[int, ...],
                    lyap_cfg: LyapunovConfig,
                    adapt_cfg: AdaptiveConfig,
                    clf_cfg: CLFConfig,
                    rollout_cfg: RolloutConfig) -> Tuple[Array, Dict[str, Array]]:
    """Roll out one trajectory and return scalar loss + summary metrics."""
    adaptive_state0 = init_adaptive_state(p, adapt_cfg)
    dt = rollout_cfg.dt

    def body(carry, _):
        x_raw, adaptive_state = carry
        x = jnp.clip(x_raw, -rollout_cfg.x_max, rollout_cfg.x_max)

        obs = make_policy_observation(x, adaptive_state, adapt_cfg)
        u_nom = policy_apply(
            params=policy_params,
            obs=obs,
            u_min=p.u_min,
            u_max=p.u_max,
            hidden_sizes=policy_hidden_sizes,
        )

        u_shield, shield_aux = clf_shield(
            u_nom=u_nom,
            x=x,
            adaptive_state=adaptive_state,
            lyap_params=lyap_params,
            lyap_cfg=lyap_cfg,
            clf_cfg=clf_cfg,
            p=p,
        )
        u = jnp.clip(u_shield, -rollout_cfg.u_clip, rollout_cfg.u_clip)

        x_next = rk4_step(acrobot_dynamics_true, x, u, a_true, dt, p)
        adaptive_next = adaptive_update_simple(adaptive_state, x, u, a_true, dt, p, adapt_cfg)

        feasible = shield_aux.get("feasible", jnp.asarray(1.0))
        step_cost = stage_cost(x, u, u_nom, feasible, rollout_cfg)

        history = {
            "x": x,
            "u": u,
            "u_nom": u_nom,
            "V": shield_aux["V"],
            "LgV": shield_aux["LgV"],
            "LyV": shield_aux["LyV"],
            "a_hat": adaptive_state.a_hat,
            "radius": adaptive_state.radius,
            "feasible": feasible,
            "step_cost": step_cost,
        }
        return (x_next, adaptive_next), history

    (xT, adaptive_T), hist = jax.lax.scan(
        body,
        init=(x0, adaptive_state0),
        xs=jnp.arange(rollout_cfg.horizon),
    )

    Q_term = jnp.asarray(rollout_cfg.Q_track)
    terminal_cost = rollout_cfg.Q_terminal_scale * (xT @ Q_term @ xT)

    mean_stage = jnp.mean(hist["step_cost"])
    total_loss = mean_stage + terminal_cost

    metrics = {
        "loss": total_loss,
        "terminal_norm": jnp.linalg.norm(xT),
        "mean_abs_proj": jnp.mean(jnp.abs(hist["u"] - hist["u_nom"])),
        "mean_feasible": jnp.mean(hist["feasible"]),
        "mean_radius": jnp.mean(hist["radius"]),
        "final_radius": adaptive_T.radius,
        "final_a_hat": adaptive_T.a_hat,
    }
    return total_loss, metrics


def batched_rollout_loss(policy_params: Dict[str, Any],
                         lyap_params: Dict[str, Any],
                         batch_x0: Array,
                         batch_a_true: Array,
                         p: AcrobotParams,
                         policy_hidden_sizes: Tuple[int, ...],
                         lyap_cfg: LyapunovConfig,
                         adapt_cfg: AdaptiveConfig,
                         clf_cfg: CLFConfig,
                         rollout_cfg: RolloutConfig) -> Tuple[Array, Dict[str, Array]]:
    rollout_fn = lambda x0, a_true: episode_rollout(
        policy_params=policy_params,
        lyap_params=lyap_params,
        x0=x0,
        a_true=a_true,
        p=p,
        policy_hidden_sizes=policy_hidden_sizes,
        lyap_cfg=lyap_cfg,
        adapt_cfg=adapt_cfg,
        clf_cfg=clf_cfg,
        rollout_cfg=rollout_cfg,
    )
    losses, metrics = jax.vmap(rollout_fn)(batch_x0, batch_a_true)
    mean_metrics = {k: jnp.mean(v) for k, v in metrics.items()}
    return jnp.mean(losses), mean_metrics


def sample_local_batch(key: Array,
                       batch_size: int,
                       p: AcrobotParams,
                       x_low: Array | None = None,
                       x_high: Array | None = None) -> Tuple[Array, Array]:
    """Sample local initial conditions and unknown parameter values."""
    if x_low is None:
        x_low = jnp.array([-0.20, -0.30, -0.75, -0.75], dtype=jnp.float32)
    if x_high is None:
        x_high = jnp.array([0.20, 0.30, 0.75, 0.75], dtype=jnp.float32)

    kx, ka = jax.random.split(key)
    batch_x0 = jax.random.uniform(kx, (batch_size, 4), minval=x_low, maxval=x_high)
    batch_a = jax.random.uniform(ka, (batch_size,), minval=p.a_min, maxval=p.a_max)
    return batch_x0, batch_a


def sample_swingup_batch(key: Array,
                         batch_size: int,
                         p: AcrobotParams,
                         dq1_range: float = 0.2,
                         dq2_range: float = 0.2,
                         w_range: float = 0.3,
                         a_true: float = 0.0) -> Tuple[Array, Array]:
    """Sample initial conditions near the hang-down position for swing-up.

    In upright coordinates hang-down is delta_q1 = -pi, delta_q2 = 0.
    """
    kx = key
    low = jnp.array([-jnp.pi - dq1_range, -dq2_range, -w_range, -w_range])
    high = jnp.array([-jnp.pi + dq1_range, dq2_range, w_range, w_range])
    batch_x0 = jax.random.uniform(kx, (batch_size, 4), minval=low, maxval=high)
    batch_a = jnp.full((batch_size,), a_true)
    return batch_x0, batch_a


def sample_curriculum_batch(key: Array,
                            batch_size: int,
                            p: AcrobotParams,
                            progress: float,
                            a_true: float = 0.0) -> Tuple[Array, Array]:
    """Sample initial conditions with curriculum: progress 0 = near upright, 1 = hang-down.

    At progress=0: x0 ~ [0 +/- 0.3, 0 +/- 0.2, +/- 0.5, +/- 0.5]
    At progress=1: x0 ~ [-pi +/- 2.8, 0 +/- 0.5, +/- 1.5, +/- 1.5]
    """
    dq1_center = -jnp.pi * progress
    dq1_range = 0.3 + 2.5 * progress
    dq2_range = 0.2 + 0.3 * progress
    w_range = 0.5 + 1.0 * progress

    low = jnp.array([dq1_center - dq1_range, -dq2_range, -w_range, -w_range])
    high = jnp.array([dq1_center + dq1_range, dq2_range, w_range, w_range])
    batch_x0 = jax.random.uniform(key, (batch_size, 4), minval=low, maxval=high)
    batch_a = jnp.full((batch_size,), a_true)
    return batch_x0, batch_a


def default_experiment_setup(key: Array,
                             p: AcrobotParams,
                             P_lqr: Array,
                             policy_hidden_sizes: Tuple[int, ...] = (64, 64)
                             ) -> Tuple[Dict[str, Any], Dict[str, Any], LyapunovConfig]:
    """Create default policy and fixed-quadratic CLF parameters.

    Observation dimension = state_dim + 2 = [x, a_hat, radius].
    """
    k_policy, k_lyap = jax.random.split(key)

    obs_dim = 4 + 2
    policy_params = init_policy_params(k_policy, obs_dim=obs_dim, hidden_sizes=policy_hidden_sizes)

    lyap_cfg = LyapunovConfig(
        mode="quadratic_fixed",
        state_dim=4,
        x_eq=(0.0, 0.0, 0.0, 0.0),
        P_init=P_lqr,
    )
    lyap_params = init_lyapunov_params(k_lyap, lyap_cfg)
    return policy_params, lyap_params, lyap_cfg


def value_and_grad_loss(policy_params: Dict[str, Any],
                        lyap_params: Dict[str, Any],
                        batch_x0: Array,
                        batch_a_true: Array,
                        p: AcrobotParams,
                        policy_hidden_sizes: Tuple[int, ...],
                        lyap_cfg: LyapunovConfig,
                        adapt_cfg: AdaptiveConfig,
                        clf_cfg: CLFConfig,
                        rollout_cfg: RolloutConfig):
    """Convenience wrapper for differentiating the batched objective."""
    def loss_fn(policy_tree: Dict[str, Any]) -> Tuple[Array, Dict[str, Array]]:
        return batched_rollout_loss(
            policy_params=policy_tree,
            lyap_params=lyap_params,
            batch_x0=batch_x0,
            batch_a_true=batch_a_true,
            p=p,
            policy_hidden_sizes=policy_hidden_sizes,
            lyap_cfg=lyap_cfg,
            adapt_cfg=adapt_cfg,
            clf_cfg=clf_cfg,
            rollout_cfg=rollout_cfg,
        )

    return jax.value_and_grad(loss_fn, has_aux=True)(policy_params)
