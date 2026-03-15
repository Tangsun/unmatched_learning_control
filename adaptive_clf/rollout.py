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
from .acrobot import acrobot_affine_terms, acrobot_dynamics_true, acrobot_energy, rk4_step
from .lyapunov import init_lyapunov_params
from .shield import clf_shield
from .adaptive import init_adaptive_state, adaptive_update_simple, make_policy_observation


def lqr_blend_control(
    u_nn: Array,
    x: Array,
    K_lqr: Array,
    P_lqr: Array,
    V_threshold: float = 5.0,
    temperature: float = 1.0,
    u_min: float = -100.0,
    u_max: float = 100.0,
) -> Array:
    """Smooth blend of NN control with LQR near upright equilibrium.

    alpha → 1 (use LQR) when V(x) < V_threshold,
    alpha → 0 (use NN)  when V(x) > V_threshold.
    """
    u_lqr = -(K_lqr @ x).squeeze()
    u_lqr = jnp.clip(u_lqr, u_min, u_max)
    V = x @ P_lqr @ x
    alpha = jax.nn.sigmoid((V_threshold - V) / jnp.maximum(temperature, 1e-6))
    return alpha * u_lqr + (1.0 - alpha) * u_nn


def stage_cost(x: Array,
               u: Array,
               u_nom: Array,
               feasible: Array,
               rollout_cfg: RolloutConfig,
               p: AcrobotParams | None = None) -> Array:
    Q = jnp.asarray(rollout_cfg.Q_track)
    x_cost = x @ Q @ x
    cost = (
        x_cost
        + rollout_cfg.R_u * (u ** 2)
        + rollout_cfg.R_proj * ((u - u_nom) ** 2)
        + rollout_cfg.R_infeasible * (1.0 - feasible)
    )
    if rollout_cfg.R_energy > 0.0 and p is not None:
        KE, PE, E_up = acrobot_energy(x, p)
        energy_err = (KE + PE) - E_up
        cost = cost + rollout_cfg.R_energy * energy_err ** 2
    return cost


def episode_rollout(policy_params: Dict[str, Any],
                    lyap_params: Dict[str, Any],
                    x0: Array,
                    a_true: Array,
                    p: AcrobotParams,
                    policy_hidden_sizes: Tuple[int, ...],
                    lyap_cfg: LyapunovConfig,
                    adapt_cfg: AdaptiveConfig,
                    clf_cfg: CLFConfig,
                    rollout_cfg: RolloutConfig,
                    K_lqr: Array | None = None,
                    P_lqr: Array | None = None) -> Tuple[Array, Dict[str, Array]]:
    """Roll out one trajectory and return scalar loss + summary metrics."""
    adaptive_state0 = init_adaptive_state(p, adapt_cfg)
    dt = rollout_cfg.dt

    bptt = rollout_cfg.bptt_window

    def body(carry, step_idx):
        x_raw, adaptive_state = carry
        if bptt > 0:
            x_raw = jnp.where(
                (step_idx > 0) & (step_idx % bptt == 0),
                jax.lax.stop_gradient(x_raw),
                x_raw,
            )
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
            affine_terms_fn=acrobot_affine_terms,
            input_bounds=(p.u_min, p.u_max),
        )
        u = jnp.clip(u_shield, -rollout_cfg.u_clip, rollout_cfg.u_clip)

        if K_lqr is not None and P_lqr is not None and rollout_cfg.lqr_blend:
            u = lqr_blend_control(
                u, x, K_lqr, P_lqr,
                V_threshold=rollout_cfg.lqr_V_threshold,
                temperature=rollout_cfg.lqr_temperature,
                u_min=p.u_min, u_max=p.u_max,
            )

        if rollout_cfg.n_substeps > 1 and K_lqr is not None and P_lqr is not None and rollout_cfg.lqr_blend:
            sub_dt = dt / rollout_cfg.n_substeps
            def _sub_body(x_sub, _):
                u_sub = lqr_blend_control(
                    u, x_sub, K_lqr, P_lqr,
                    V_threshold=rollout_cfg.lqr_V_threshold,
                    temperature=rollout_cfg.lqr_temperature,
                    u_min=p.u_min, u_max=p.u_max,
                )
                x_next_sub = rk4_step(
                    acrobot_dynamics_true, x_sub, u_sub, a_true, sub_dt, p)
                return x_next_sub, None
            x_next_raw, _ = jax.lax.scan(
                _sub_body, x, None, length=rollout_cfg.n_substeps)
        else:
            x_next_raw = rk4_step(
                acrobot_dynamics_true, x, u, a_true, dt, p,
                n_substeps=rollout_cfg.n_substeps)
        x_next = jnp.clip(
            jnp.nan_to_num(x_next_raw, nan=0.0),
            -rollout_cfg.x_max, rollout_cfg.x_max,
        )
        adaptive_next = (
            adaptive_update_simple(adaptive_state, x, u, a_true, dt, p, adapt_cfg,
                                   affine_terms_fn=acrobot_affine_terms,
                                   dynamics_fn=acrobot_dynamics_true)
            if adapt_cfg.adapt_enabled else adaptive_state
        )

        feasible = shield_aux.get("feasible", jnp.asarray(1.0))
        step_cost = stage_cost(x, u, u_nom, feasible, rollout_cfg, p)

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

    xT_safe = jnp.nan_to_num(xT, nan=rollout_cfg.x_max)
    Q_term = jnp.asarray(rollout_cfg.Q_track)
    terminal_cost = rollout_cfg.Q_terminal_scale * (xT_safe @ Q_term @ xT_safe)

    mean_stage = jnp.mean(hist["step_cost"])
    total_loss = jnp.nan_to_num(mean_stage + terminal_cost, nan=1e4)

    metrics = {
        "loss": total_loss,
        "terminal_norm": jnp.linalg.norm(xT_safe),
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
                         rollout_cfg: RolloutConfig,
                         K_lqr: Array | None = None,
                         P_lqr: Array | None = None) -> Tuple[Array, Dict[str, Array]]:
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
        K_lqr=K_lqr,
        P_lqr=P_lqr,
    )
    losses, metrics = jax.vmap(rollout_fn)(batch_x0, batch_a_true)
    safe_losses = jnp.nan_to_num(losses, nan=1e4)
    mean_metrics = {k: jnp.nan_to_num(jnp.mean(v), nan=0.0)
                    for k, v in metrics.items()}
    return jnp.mean(safe_losses), mean_metrics


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


def sample_uniform_batch(key: Array,
                         batch_size: int,
                         p: AcrobotParams,
                         dq1_range: float = jnp.pi,
                         dq2_range: float = jnp.pi,
                         w_range: float = 2.0,
                         a_true: float = 0.0) -> Tuple[Array, Array]:
    """Sample initial conditions uniformly across the state space."""
    low = jnp.array([-dq1_range, -dq2_range, -w_range, -w_range])
    high = jnp.array([dq1_range, dq2_range, w_range, w_range])
    batch_x0 = jax.random.uniform(key, (batch_size, 4), minval=low, maxval=high)
    batch_a = jnp.full((batch_size,), a_true)
    return batch_x0, batch_a


def sample_curriculum_batch(key: Array,
                            batch_size: int,
                            p: AcrobotParams,
                            progress: float,
                            a_true: float = 0.0,
                            region_scale: float = 1.0) -> Tuple[Array, Array]:
    """Sample initial conditions with curriculum: progress 0 = near upright, 1 = hang-down.

    At progress=0: x0 ~ [0 +/- 0.3, 0 +/- 0.2, +/- 0.5, +/- 0.5]
    At progress=1: x0 ~ [-pi +/- 2.8, 0 +/- 0.5, +/- 1.5, +/- 1.5]
    region_scale multiplies the ranges.
    """
    dq1_center = -jnp.pi * progress
    dq1_range = region_scale * (0.3 + 2.5 * progress)
    dq2_range = region_scale * (0.2 + 0.3 * progress)
    w_range = region_scale * (0.5 + 1.0 * progress)

    low = jnp.array([dq1_center - dq1_range, -dq2_range, -w_range, -w_range])
    high = jnp.array([dq1_center + dq1_range, dq2_range, w_range, w_range])
    batch_x0 = jax.random.uniform(key, (batch_size, 4), minval=low, maxval=high)
    batch_a = jnp.full((batch_size,), a_true)
    return batch_x0, batch_a


def sample_mixed_batch(key: Array,
                       batch_size: int,
                       p: AcrobotParams,
                       region_scale: float = 1.0,
                       a_true: float = 0.0) -> Tuple[Array, Array]:
    """Sample ICs from a geometric mixture of scales up to region_scale.

    Half the batch comes from the full region, the other half from
    geometrically smaller regions (keeping the policy sharp near upright).
    """
    n_scales = 4
    base_scale = max(region_scale / (3 ** (n_scales - 1)), 0.001)
    scales = [base_scale * (3 ** i) for i in range(n_scales)]
    scales = [min(s, region_scale) for s in scales]

    per_scale = batch_size // n_scales
    remainder = batch_size - per_scale * n_scales
    keys = jax.random.split(key, n_scales)

    batches = []
    for i, (k, s) in enumerate(zip(keys, scales)):
        n = per_scale + (1 if i < remainder else 0)
        dq1_r = float(s * jnp.pi)
        dq2_r = float(s * jnp.pi)
        w_r = float(s * 2.0)
        low = jnp.array([-dq1_r, -dq2_r, -w_r, -w_r])
        high = jnp.array([dq1_r, dq2_r, w_r, w_r])
        batches.append(jax.random.uniform(k, (n, 4), minval=low, maxval=high))

    batch_x0 = jnp.concatenate(batches, axis=0)
    batch_a = jnp.full((batch_size,), a_true)
    return batch_x0, batch_a


def default_experiment_setup(key: Array,
                             p: AcrobotParams,
                             P_lqr: Array,
                             policy_hidden_sizes: Tuple[int, ...] = (64, 64),
                             adapt_cfg: AdaptiveConfig | None = None,
                             ) -> Tuple[Dict[str, Any], Dict[str, Any], LyapunovConfig]:
    """Create default policy and fixed-quadratic CLF parameters.

    Observation: x (4D) when adapt_enabled=False, else [x, a_hat, radius] (6D).
    """
    if adapt_cfg is None:
        adapt_cfg = AdaptiveConfig()
    k_policy, k_lyap = jax.random.split(key)

    obs_dim = 4 if not adapt_cfg.adapt_enabled else 4 + 2
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
                        rollout_cfg: RolloutConfig,
                        K_lqr: Array | None = None,
                        P_lqr: Array | None = None):
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
            K_lqr=K_lqr,
            P_lqr=P_lqr,
        )

    return jax.value_and_grad(loss_fn, has_aux=True)(policy_params)


def value_and_grad_joint(all_params: Dict[str, Any],
                         batch_x0: Array,
                         batch_a_true: Array,
                         p: AcrobotParams,
                         policy_hidden_sizes: Tuple[int, ...],
                         lyap_cfg: LyapunovConfig,
                         adapt_cfg: AdaptiveConfig,
                         clf_cfg: CLFConfig,
                         rollout_cfg: RolloutConfig,
                         K_lqr: Array | None = None,
                         P_lqr: Array | None = None):
    """Differentiate batched loss w.r.t. joint (policy, lyapunov) params."""
    def loss_fn(params: Dict[str, Any]) -> Tuple[Array, Dict[str, Array]]:
        return batched_rollout_loss(
            policy_params=params["policy"],
            lyap_params=params["lyap"],
            batch_x0=batch_x0,
            batch_a_true=batch_a_true,
            p=p,
            policy_hidden_sizes=policy_hidden_sizes,
            lyap_cfg=lyap_cfg,
            adapt_cfg=adapt_cfg,
            clf_cfg=clf_cfg,
            rollout_cfg=rollout_cfg,
            K_lqr=K_lqr,
            P_lqr=P_lqr,
        )

    return jax.value_and_grad(loss_fn, has_aux=True)(all_params)
