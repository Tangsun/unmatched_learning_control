"""Dubins car path-following: train NN policy with adaptive CLF shield.

Self-contained training script.  Does NOT modify nn.py or other shared
modules -- all multi-output policy logic lives here.

Usage:
    python -m adaptive_clf.train_dubins --epochs 200
    python -m adaptive_clf.train_dubins --clf --epochs 200
    python -m adaptive_clf.train_dubins --adapt --clf --epochs 200
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from .configs import Array, AdaptiveConfig, AdaptiveState, CLFConfig, LyapunovConfig, MLPConfig
from .nn import init_mlp_params, apply_mlp
from .dubins import (
    DubinsParams, dubins_affine_terms, dubins_dynamics,
    rk4_step_dubins, solve_dubins_lqr,
)
from .lyapunov import init_lyapunov_params, lyapunov_value_and_grad
from .shield import clf_shield


# ---------------------------------------------------------------------------
# Multi-output policy helpers  (local to this file)
# ---------------------------------------------------------------------------

STATE_DIM = 3   # [e_x, e_y, e_theta]
CTRL_DIM = 2    # [v, omega]


def _init_policy(key: Array, obs_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64)) -> Dict[str, Any]:
    cfg = MLPConfig(
        in_dim=obs_dim,
        hidden_sizes=hidden_sizes,
        out_dim=CTRL_DIM,
        activation="tanh",
        final_activation="identity",
        output_scale=1.0,
    )
    return init_mlp_params(key, cfg)


def _policy_apply(params: Dict[str, Any], obs: Array,
                  p: DubinsParams,
                  hidden_sizes: Tuple[int, ...] = (64, 64)) -> Array:
    """Map observation -> [v, omega] with per-channel tanh scaling."""
    cfg = MLPConfig(
        in_dim=obs.shape[-1],
        hidden_sizes=hidden_sizes,
        out_dim=CTRL_DIM,
        activation="tanh",
        final_activation="identity",
        output_scale=1.0,
    )
    raw = apply_mlp(params, obs, cfg)          # (2,)
    # Scale each channel independently through tanh
    u_mid = jnp.array([0.5 * (p.v_max + p.v_min),
                        0.5 * (p.omega_max + p.omega_min)])
    u_half = jnp.array([0.5 * (p.v_max - p.v_min),
                         0.5 * (p.omega_max - p.omega_min)])
    return u_mid + u_half * jnp.tanh(raw)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def _make_obs(e: Array, adaptive_state: AdaptiveState | None,
              adapt_enabled: bool) -> Array:
    """Build observation vector from error state (+ optional adaptive info)."""
    # Use sin/cos for heading to avoid discontinuity
    obs = jnp.array([e[0], e[1], jnp.sin(e[2]), jnp.cos(e[2]) - 1.0])
    if adapt_enabled and adaptive_state is not None:
        obs = jnp.concatenate([obs, jnp.array([adaptive_state.a_hat,
                                                 adaptive_state.radius])])
    return obs


def _obs_dim(adapt_enabled: bool) -> int:
    return 6 if adapt_enabled else 4


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DubinsCost:
    w_ex: float = 0.5       # along-track error
    w_ey: float = 5.0       # lateral error (most important)
    w_eth: float = 2.0      # heading error
    w_v: float = 0.01       # speed control effort
    w_omega: float = 0.01   # turn rate effort
    w_terminal: float = 10.0
    w_proj: float = 0.1     # CLF projection penalty
    w_infeasible: float = 50.0


def _stage_cost(e: Array, u: Array, u_nom: Array, feasible: Array,
                cfg: DubinsCost) -> Array:
    cost = (
        cfg.w_ex * e[0] ** 2
        + cfg.w_ey * e[1] ** 2
        + cfg.w_eth * e[2] ** 2
        + cfg.w_v * u[0] ** 2
        + cfg.w_omega * u[1] ** 2
        + cfg.w_proj * jnp.sum((u - u_nom) ** 2)
        + cfg.w_infeasible * (1.0 - feasible)
    )
    return cost


def _terminal_cost(eT: Array, cfg: DubinsCost) -> Array:
    return cfg.w_terminal * (
        cfg.w_ex * eT[0] ** 2
        + cfg.w_ey * eT[1] ** 2
        + cfg.w_eth * eT[2] ** 2
    )


# ---------------------------------------------------------------------------
# Adaptive update  (simple online estimator, scalar a)
# ---------------------------------------------------------------------------

def _adaptive_update(state: AdaptiveState,
                     e: Array, u: Array, a_true: Array,
                     dt: float, p: DubinsParams,
                     adapt_cfg: AdaptiveConfig) -> AdaptiveState:
    """Online estimator for unknown side-slip."""
    f, g, y = dubins_affine_terms(e, p)
    edot_true = dubins_dynamics(e, u, a_true, p)
    edot_nom = f + g @ u + y * state.a_hat
    residual = edot_true - edot_nom          # = y * (a_true - a_hat)

    # Gradient step on a_hat
    a_hat_next = state.a_hat + adapt_cfg.eta * jnp.dot(y, residual) * dt
    a_hat_next = jnp.clip(a_hat_next, p.a_min, p.a_max)

    # Information accumulation -> shrinking radius
    info_next = state.info + jnp.dot(y, y) * dt
    radius_next = adapt_cfg.radius_scale / jnp.sqrt(info_next + 1e-6)
    radius_next = jnp.maximum(radius_next, adapt_cfg.radius_floor)

    return AdaptiveState(a_hat=a_hat_next, info=info_next, radius=radius_next)


def _init_adaptive(p: DubinsParams, adapt_cfg: AdaptiveConfig) -> AdaptiveState:
    if not adapt_cfg.adapt_enabled:
        return AdaptiveState(
            a_hat=jnp.array(0.0), info=jnp.array(1e-6), radius=jnp.array(0.0))
    a_hat0 = jnp.float32(0.5 * (p.a_min + p.a_max))
    info0 = jnp.float32(adapt_cfg.info_init)
    radius0 = jnp.float32(max(0.5 * (p.a_max - p.a_min), adapt_cfg.radius_floor))
    return AdaptiveState(a_hat=a_hat0, info=info0, radius=radius0)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _episode_rollout(
    policy_params: Dict[str, Any],
    lyap_params: Dict[str, Any] | None,
    e0: Array,
    a_true: Array,
    p: DubinsParams,
    hidden_sizes: Tuple[int, ...],
    lqr_K: Array,
    horizon: int,
    dt: float,
    cost_cfg: DubinsCost,
    clf_cfg: CLFConfig | None,
    lyap_cfg: LyapunovConfig | None,
    adapt_cfg: AdaptiveConfig,
    use_lqr_blend: bool,
    lqr_blend_V_thresh: float,
) -> Tuple[Array, Dict[str, Array]]:
    e_max = 20.0
    use_clf = (clf_cfg is not None and clf_cfg.enabled and lyap_params is not None)
    use_adapt = adapt_cfg.adapt_enabled

    adaptive_state0 = _init_adaptive(p, adapt_cfg)
    u_eq = jnp.array([p.v_ref, 0.0])

    def body(carry, _):
        e_raw, adaptive_state = carry
        e = jnp.clip(e_raw, -e_max, e_max)

        # Nominal policy
        obs = _make_obs(e, adaptive_state, use_adapt)
        u_nom = _policy_apply(policy_params, obs, p, hidden_sizes)

        # Optional LQR blending: LQR dominates when V(e) is small
        if use_lqr_blend:
            u_lqr_delta = -lqr_K @ e                         # (2,)
            u_lqr = u_eq + u_lqr_delta
            u_lqr = jnp.array([
                jnp.clip(u_lqr[0], p.v_min, p.v_max),
                jnp.clip(u_lqr[1], p.omega_min, p.omega_max),
            ])
            if lyap_params is not None and lyap_cfg is not None:
                V_val, _ = lyapunov_value_and_grad(lyap_params, lyap_cfg, e)
            else:
                V_val = e @ e  # fallback
            alpha = jax.nn.sigmoid((lqr_blend_V_thresh - V_val) / 0.5)
            u_nom = alpha * u_lqr + (1.0 - alpha) * u_nom

        # CLF shield
        if use_clf:
            u_shield, shield_aux = clf_shield(
                u_nom=u_nom,
                x=e,
                adaptive_state=adaptive_state,
                lyap_params=lyap_params,
                lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg,
                p=p,
                affine_terms_fn=dubins_affine_terms,
            )
            # Clip to input bounds
            u = jnp.array([
                jnp.clip(u_shield[0], p.v_min, p.v_max),
                jnp.clip(u_shield[1], p.omega_min, p.omega_max),
            ])
            feasible = shield_aux.get("feasible", jnp.array(1.0))
            V_val = shield_aux["V"]
        else:
            u = u_nom
            feasible = jnp.array(1.0)
            V_val = jnp.array(0.0)

        # Adaptive update
        if use_adapt:
            adaptive_next = _adaptive_update(
                adaptive_state, e, u, a_true, dt, p, adapt_cfg)
        else:
            adaptive_next = adaptive_state

        # Step dynamics
        e_next = rk4_step_dubins(e, u, a_true, dt, p)
        e_next = jnp.nan_to_num(e_next, nan=0.0)

        cost = _stage_cost(e, u, u_nom, feasible, cost_cfg)
        history = {
            "e": e, "u": u, "u_nom": u_nom, "cost": cost,
            "V": V_val, "feasible": feasible,
            "a_hat": adaptive_state.a_hat, "radius": adaptive_state.radius,
        }
        return (e_next, adaptive_next), history

    (eT, _), hist = jax.lax.scan(body, (e0, adaptive_state0), jnp.arange(horizon))
    eT = jnp.nan_to_num(eT, nan=e_max)

    mean_stage = jnp.mean(hist["cost"])
    terminal = _terminal_cost(eT, cost_cfg)
    total_loss = jnp.nan_to_num(mean_stage + terminal, nan=1e4)

    metrics = {
        "loss": total_loss,
        "terminal_norm": jnp.linalg.norm(eT),
        "mean_stage_cost": mean_stage,
        "terminal_cost": terminal,
        "mean_feasible": jnp.mean(hist["feasible"]),
        "mean_V": jnp.mean(hist["V"]),
        "mean_proj": jnp.mean(jnp.sqrt(jnp.sum((hist["u"] - hist["u_nom"]) ** 2, axis=-1))),
    }
    return total_loss, metrics


def _batched_loss(policy_params, lyap_params, batch_e0, batch_a, p,
                  hidden_sizes, lqr_K, horizon, dt, cost_cfg,
                  clf_cfg, lyap_cfg, adapt_cfg, use_lqr_blend,
                  lqr_blend_V_thresh):
    fn = lambda e0, a: _episode_rollout(
        policy_params, lyap_params, e0, a, p, hidden_sizes, lqr_K,
        horizon, dt, cost_cfg, clf_cfg, lyap_cfg, adapt_cfg,
        use_lqr_blend, lqr_blend_V_thresh,
    )
    losses, metrics = jax.vmap(fn)(batch_e0, batch_a)
    safe = jnp.nan_to_num(losses, nan=1e4)
    return jnp.mean(safe), {k: jnp.nan_to_num(jnp.mean(v), nan=0.0)
                             for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_batch(key: Array, batch_size: int, p: DubinsParams,
                  region_scale: float = 1.0,
                  a_range: bool = True) -> Tuple[Array, Array]:
    """Sample initial error states and unknown side-slip values."""
    ke, ka = jax.random.split(key)
    low = jnp.array([-1.0, -1.0, -0.8]) * region_scale
    high = jnp.array([1.0, 1.0, 0.8]) * region_scale
    batch_e0 = jax.random.uniform(ke, (batch_size, STATE_DIM), minval=low, maxval=high)
    if a_range:
        batch_a = jax.random.uniform(ka, (batch_size,),
                                      minval=p.a_min, maxval=p.a_max)
    else:
        batch_a = jnp.zeros(batch_size)
    return batch_e0, batch_a


def _sample_curriculum_batch(key: Array, batch_size: int, p: DubinsParams,
                             progress: float,
                             region_scale: float = 1.0,
                             a_range: bool = True) -> Tuple[Array, Array]:
    """Curriculum: 50 % small-error ICs + 50 % expanding ICs."""
    k_small, k_cur, ka = jax.random.split(key, 3)
    n_small = max(1, batch_size // 2)
    n_cur = batch_size - n_small

    # Small-error ICs (always present to prevent forgetting)
    small_low = jnp.array([-0.2, -0.2, -0.2]) * region_scale
    small_high = jnp.array([0.2, 0.2, 0.2]) * region_scale
    batch_small = jax.random.uniform(k_small, (n_small, STATE_DIM),
                                      minval=small_low, maxval=small_high)

    # Expanding ICs
    scale = 0.2 + 0.8 * progress  # 0.2 -> 1.0
    cur_low = jnp.array([-1.0, -1.0, -0.8]) * region_scale * scale
    cur_high = jnp.array([1.0, 1.0, 0.8]) * region_scale * scale
    batch_cur = jax.random.uniform(k_cur, (n_cur, STATE_DIM),
                                    minval=cur_low, maxval=cur_high)

    batch_e0 = jnp.concatenate([batch_small, batch_cur], axis=0)
    if a_range:
        batch_a = jax.random.uniform(ka, (batch_size,),
                                      minval=p.a_min, maxval=p.a_max)
    else:
        batch_a = jnp.zeros(batch_size)
    return batch_e0, batch_a


# ---------------------------------------------------------------------------
# Sanitize gradients
# ---------------------------------------------------------------------------

def _sanitize_grads(grads):
    return jax.tree.map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_dubins(
    epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 128,
    pool_size: int = 2048,
    horizon: int = 200,
    dt: float = 0.02,
    max_grad_norm: float = 10.0,
    curriculum_frac: float = 0.5,
    log_every: int = 5,
    save_dir: str | None = None,
    seed: int = 0,
    # CLF
    clf_enabled: bool = True,
    lambda_clf: float = 0.5,
    region_scale: float = 1.0,
    # Adaptive
    adapt_enabled: bool = False,
    adapt_eta: float = 0.1,
    adapt_radius_scale: float = 1.0,
    # LQR blend
    use_lqr_blend: bool = False,
    lqr_blend_V_thresh: float = 2.0,
    # DubinsParams overrides
    v_ref: float = 1.0,
    a_min: float = -0.5,
    a_max: float = 0.5,
) -> Tuple[Dict[str, Any], list]:

    p = DubinsParams(v_ref=v_ref, a_min=a_min, a_max=a_max)

    # LQR
    Q_lqr = jnp.diag(jnp.array([1.0, 10.0, 5.0]))
    R_lqr = jnp.diag(jnp.array([0.1, 0.1]))
    P_lqr, lqr_K = solve_dubins_lqr(p, Q_lqr, R_lqr)
    print(f"LQR gain K =\n{lqr_K}")
    print(f"P eigenvalues: {jnp.linalg.eigvalsh(P_lqr)}")

    # Cost
    cost_cfg = DubinsCost()

    # CLF / Lyapunov
    clf_cfg = CLFConfig(enabled=clf_enabled, lambda_clf=lambda_clf)
    lyap_cfg = LyapunovConfig(
        mode="quadratic_fixed",
        state_dim=STATE_DIM,
        x_eq=(0.0, 0.0, 0.0),
        P_init=P_lqr,
    )
    key = jax.random.PRNGKey(seed)
    key, lyap_key = jax.random.split(key)
    lyap_params = init_lyapunov_params(lyap_key, lyap_cfg)

    # Adaptive
    adapt_cfg = AdaptiveConfig(
        adapt_enabled=adapt_enabled,
        eta=adapt_eta,
        radius_scale=adapt_radius_scale,
        radius_floor=1e-3,
        info_init=1e-3,
    )

    # Policy
    hidden_sizes = (64, 64)
    obs_dim = _obs_dim(adapt_enabled)
    key, init_key = jax.random.split(key)
    policy_params = _init_policy(init_key, obs_dim, hidden_sizes)

    # Save dir
    mode_tag = "clf" if clf_enabled else "nn"
    if adapt_enabled:
        mode_tag += "_adapt"
    if use_lqr_blend:
        mode_tag += "_lqr"
    if save_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        save_dir = (f"runs/dubins_{mode_tag}_{ts}_ep{epochs}_h{horizon}"
                    f"_pool{pool_size}_bs{batch_size}_lr{lr}")
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer
    steps_per_epoch = pool_size // batch_size
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )
    opt_state = optimizer.init(policy_params)

    use_curriculum = region_scale >= 0.5
    curriculum_end = max(1, int(curriculum_frac * epochs)) if use_curriculum else 0

    @jax.jit
    def step_fn(params, opt_state, batch_e0, batch_a):
        def loss_fn(pp):
            return _batched_loss(
                pp, lyap_params, batch_e0, batch_a, p, hidden_sizes, lqr_K,
                horizon, dt, cost_cfg, clf_cfg, lyap_cfg, adapt_cfg,
                use_lqr_blend, lqr_blend_V_thresh,
            )
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        grads = _sanitize_grads(grads)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        metrics["grad_norm"] = grad_norm
        return new_params, new_opt, loss, metrics

    # Print config
    total_steps = epochs * steps_per_epoch
    print(f"\n{'='*60}")
    print(f"  Dubins path-following | mode={mode_tag}")
    print(f"  {epochs} epochs, horizon={horizon}, dt={dt}")
    print(f"  pool={pool_size}, batch={batch_size}, "
          f"{steps_per_epoch} steps/epoch, {total_steps} total")
    print(f"  lr={lr}, grad_clip={max_grad_norm}")
    print(f"  CLF: {'ON' if clf_enabled else 'OFF'} (lambda={lambda_clf})")
    print(f"  Adaptive: {'ON' if adapt_enabled else 'OFF'}")
    print(f"  Side-slip range: [{a_min}, {a_max}]")
    print(f"  Region scale: {region_scale}")
    print(f"  Saving to: {save_dir}")
    print(f"{'='*60}\n")

    history = []
    t0 = time.time()
    global_step = 0
    best_loss = float("inf")
    best_params = policy_params

    for epoch in range(1, epochs + 1):
        key, pool_key = jax.random.split(key)

        if use_curriculum:
            cur_progress = min(1.0, (epoch - 1) / max(1, curriculum_end - 1))
            pool_e0, pool_a = _sample_curriculum_batch(
                pool_key, pool_size, p, cur_progress,
                region_scale=region_scale, a_range=(a_max > 0),
            )
            phase = f"cur {cur_progress:.2f}" if cur_progress < 1.0 else "full"
        else:
            pool_e0, pool_a = _sample_batch(
                pool_key, pool_size, p,
                region_scale=region_scale, a_range=(a_max > 0),
            )
            phase = f"r={region_scale:.1f}"

        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, pool_size)
        pool_e0 = pool_e0[perm]
        pool_a = pool_a[perm]

        epoch_loss = 0.0
        epoch_grad = 0.0
        epoch_term = 0.0
        epoch_feas = 0.0

        for i in range(steps_per_epoch):
            global_step += 1
            start = i * batch_size
            batch_e0 = pool_e0[start:start + batch_size]
            batch_a = pool_a[start:start + batch_size]

            policy_params, opt_state, loss, metrics = step_fn(
                policy_params, opt_state, batch_e0, batch_a)

            loss_val = float(loss)
            epoch_loss += loss_val
            epoch_grad += float(metrics["grad_norm"])
            epoch_term += float(metrics["terminal_norm"])
            epoch_feas += float(metrics.get("mean_feasible", 1.0))

            record = {"step": global_step, "epoch": epoch, "loss": loss_val}
            for k, v in metrics.items():
                record[k] = float(v)
            history.append(record)

        n = steps_per_epoch
        avg_loss = epoch_loss / n
        avg_grad = epoch_grad / n
        avg_term = epoch_term / n
        avg_feas = epoch_feas / n

        if (not use_curriculum or epoch > curriculum_end) and avg_loss < best_loss:
            best_loss = avg_loss
            best_params = jax.tree.map(lambda x: x.copy(), policy_params)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            elapsed = time.time() - t0
            best_str = f"best {best_loss:8.4f}" if best_loss < float("inf") else "best      ---"
            feas_str = f"feas {avg_feas:.3f} | " if clf_enabled else ""
            print(
                f"epoch {epoch:4d}/{epochs} "
                f"[{phase:8s}] | "
                f"loss {avg_loss:9.4f} | "
                f"term_norm {avg_term:7.3f} | "
                f"{feas_str}"
                f"grad {avg_grad:8.4f} | "
                f"{best_str} | "
                f"{elapsed:.1f}s"
            )

    # Save
    save_all = best_params if best_loss < float("inf") else policy_params
    save_data = {
        "nn": jax.device_get(save_all),
        "lqr_K": jax.device_get(lqr_K),
        "P_lqr": jax.device_get(P_lqr),
        "clf_enabled": clf_enabled,
        "lambda_clf": lambda_clf,
        "adapt_enabled": adapt_enabled,
        "v_ref": v_ref,
        "a_min": a_min,
        "a_max": a_max,
    }

    params_path = os.path.join(save_dir, "policy_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nSaved policy -> {params_path}")

    hist_path = os.path.join(save_dir, "history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    run_config = {
        "mode": mode_tag, "clf_enabled": clf_enabled,
        "adapt_enabled": adapt_enabled, "lambda_clf": lambda_clf,
        "region_scale": region_scale, "epochs": epochs, "lr": lr,
        "batch_size": batch_size, "pool_size": pool_size,
        "horizon": horizon, "dt": dt, "max_grad_norm": max_grad_norm,
        "seed": seed, "v_ref": v_ref, "a_min": a_min, "a_max": a_max,
    }
    config_path = os.path.join(save_dir, "run_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(run_config, f)

    _plot_training(history, save_dir, clf_enabled)
    _eval_policy(save_all, p, hidden_sizes, lqr_K, horizon, dt,
                 cost_cfg, save_dir, clf_enabled, lyap_params, lyap_cfg,
                 clf_cfg, adapt_cfg, use_lqr_blend, lqr_blend_V_thresh)

    return save_data, history


# ---------------------------------------------------------------------------
# Plotting / evaluation
# ---------------------------------------------------------------------------

def _plot_training(history: list, save_dir: str, clf_enabled: bool) -> None:
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    norms = [h["terminal_norm"] for h in history]

    ncols = 3 if clf_enabled else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    axes[0].plot(steps, losses)
    axes[0].set(xlabel="step", ylabel="loss", title="Training loss")
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, norms, color="tab:orange")
    axes[1].set(xlabel="step", ylabel="|e_T|", title="Terminal error norm")
    axes[1].grid(True, alpha=0.3)

    if clf_enabled and ncols == 3:
        feas = [h.get("mean_feasible", 1.0) for h in history]
        axes[2].plot(steps, feas, color="tab:green")
        axes[2].set(xlabel="step", ylabel="feasibility", title="CLF feasibility")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(save_dir, "training_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curve -> {out}")


def _simulate(policy_params, e0, a_true, p, hidden_sizes, lqr_K,
              horizon, dt, clf_enabled, lyap_params, lyap_cfg, clf_cfg,
              adapt_cfg, use_lqr_blend, lqr_blend_V_thresh):
    """Roll out policy and collect trajectory."""
    e_max = 20.0
    use_clf = clf_enabled and lyap_params is not None
    use_adapt = adapt_cfg.adapt_enabled
    adaptive_state0 = _init_adaptive(p, adapt_cfg)
    u_eq = jnp.array([p.v_ref, 0.0])

    def body(carry, _):
        e_raw, adaptive_state = carry
        e = jnp.clip(e_raw, -e_max, e_max)

        obs = _make_obs(e, adaptive_state, use_adapt)
        u_nom = _policy_apply(policy_params, obs, p, hidden_sizes)

        if use_lqr_blend:
            u_lqr = u_eq + (-lqr_K @ e)
            u_lqr = jnp.array([
                jnp.clip(u_lqr[0], p.v_min, p.v_max),
                jnp.clip(u_lqr[1], p.omega_min, p.omega_max),
            ])
            V_val, _ = lyapunov_value_and_grad(lyap_params, lyap_cfg, e)
            alpha = jax.nn.sigmoid((lqr_blend_V_thresh - V_val) / 0.5)
            u_nom = alpha * u_lqr + (1.0 - alpha) * u_nom

        if use_clf:
            u_shield, _ = clf_shield(
                u_nom=u_nom, x=e, adaptive_state=adaptive_state,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p,
                affine_terms_fn=dubins_affine_terms,
            )
            u = jnp.array([
                jnp.clip(u_shield[0], p.v_min, p.v_max),
                jnp.clip(u_shield[1], p.omega_min, p.omega_max),
            ])
        else:
            u = u_nom

        if use_adapt:
            adaptive_next = _adaptive_update(
                adaptive_state, e, u, a_true, dt, p, adapt_cfg)
        else:
            adaptive_next = adaptive_state

        e_next = rk4_step_dubins(e, u, a_true, dt, p)
        return (e_next, adaptive_next), (e, u, adaptive_state.a_hat, adaptive_state.radius)

    _, (es, us, a_hats, radii) = jax.lax.scan(
        body, (e0, adaptive_state0), jnp.arange(horizon))
    return es, us, a_hats, radii


def _eval_policy(policy_params, p, hidden_sizes, lqr_K, horizon, dt,
                 cost_cfg, save_dir, clf_enabled, lyap_params, lyap_cfg,
                 clf_cfg, adapt_cfg, use_lqr_blend, lqr_blend_V_thresh):
    eval_horizon = max(horizon, 400)
    test_slips = [0.0, 0.3, -0.3, 0.5]
    ics = [
        (jnp.array([0.0, 0.5, 0.0]), "ey=0.5"),
        (jnp.array([0.0, -0.5, 0.3]), "ey=-0.5, eth=0.3"),
        (jnp.array([0.5, 0.8, -0.5]), "ex=0.5, ey=0.8, eth=-0.5"),
        (jnp.array([0.0, 0.0, 0.0]), "origin (test slip only)"),
    ]

    ts = jnp.arange(eval_horizon) * dt

    for a_val in test_slips:
        a_true = jnp.float32(a_val)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        ax_ey, ax_eth, ax_u, ax_adapt = axes.flat

        for e0, label in ics:
            es, us, a_hats, radii = _simulate(
                policy_params, e0, a_true, p, hidden_sizes, lqr_K,
                eval_horizon, dt, clf_enabled, lyap_params, lyap_cfg,
                clf_cfg, adapt_cfg, use_lqr_blend, lqr_blend_V_thresh,
            )
            ax_ey.plot(ts, es[:, 1], label=label)
            ax_eth.plot(ts, jnp.degrees(es[:, 2]), label=label)
            ax_u.plot(ts, us[:, 0], label=f"v: {label}", linestyle="-")
            ax_u.plot(ts, us[:, 1], label=None, linestyle="--", alpha=0.5)
            if adapt_cfg.adapt_enabled:
                ax_adapt.plot(ts, a_hats, label=label)

        ax_ey.set(ylabel="e_y", title="Lateral error")
        ax_ey.axhline(0, color="k", ls=":", lw=0.8)
        ax_eth.set(ylabel="e_theta (deg)", title="Heading error")
        ax_eth.axhline(0, color="k", ls=":", lw=0.8)
        ax_u.set(ylabel="u", title="Controls (solid=v, dashed=omega)")

        if adapt_cfg.adapt_enabled:
            ax_adapt.axhline(a_val, color="k", ls=":", lw=1.2, label=f"true a={a_val}")
            ax_adapt.set(ylabel="a_hat", title="Parameter estimate")
            ax_adapt.legend(fontsize=7)
        else:
            ax_adapt.set_visible(False)

        for ax in [ax_ey, ax_eth, ax_u]:
            ax.set_xlabel("time (s)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        fig.suptitle(f"Dubins eval | slip a={a_val:.2f} | CLF={'on' if clf_enabled else 'off'}",
                     fontsize=13)
        fig.tight_layout()
        out = os.path.join(save_dir, f"eval_slip_{a_val:.2f}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved eval plot -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dubins path-following")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--curriculum-frac", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # CLF
    parser.add_argument("--clf", action="store_true", default=False)
    parser.add_argument("--no-clf", dest="clf", action="store_false")
    parser.add_argument("--lambda-clf", type=float, default=0.5)
    parser.add_argument("--region-scale", type=float, default=1.0)
    # Adaptive
    parser.add_argument("--adapt", action="store_true", default=False)
    parser.add_argument("--adapt-eta", type=float, default=0.1)
    parser.add_argument("--adapt-radius-scale", type=float, default=1.0)
    # LQR blend
    parser.add_argument("--lqr-blend", action="store_true", default=False)
    parser.add_argument("--lqr-blend-V-thresh", type=float, default=2.0)
    # System
    parser.add_argument("--v-ref", type=float, default=1.0)
    parser.add_argument("--a-min", type=float, default=-0.5)
    parser.add_argument("--a-max", type=float, default=0.5)
    args = parser.parse_args()

    train_dubins(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        pool_size=args.pool_size, horizon=args.horizon, dt=args.dt,
        max_grad_norm=args.max_grad_norm, curriculum_frac=args.curriculum_frac,
        log_every=args.log_every, save_dir=args.save_dir, seed=args.seed,
        clf_enabled=args.clf, lambda_clf=args.lambda_clf,
        region_scale=args.region_scale, adapt_enabled=args.adapt,
        adapt_eta=args.adapt_eta, adapt_radius_scale=args.adapt_radius_scale,
        use_lqr_blend=args.lqr_blend, lqr_blend_V_thresh=args.lqr_blend_V_thresh,
        v_ref=args.v_ref, a_min=args.a_min, a_max=args.a_max,
    )


if __name__ == "__main__":
    main()
