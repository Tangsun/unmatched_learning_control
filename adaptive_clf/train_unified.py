"""Unified training script for any control-affine system.

Supports acrobot, cartpole, dubins car (extensible via systems.py registry).
Three policy modes: nn_only, hybrid (LQR+NN blend), clf (NN+CLF shield).

Usage:
    python -m adaptive_clf.train_unified --system cartpole --epochs 100
    python -m adaptive_clf.train_unified --system cartpole --clf --region-scale 0.3
    python -m adaptive_clf.train_unified --system dubins --clf --adapt --epochs 200
    python -m adaptive_clf.train_unified --system cartpole --learn-lyap --region-scale 0.3
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

from .configs import Array, AdaptiveConfig, AdaptiveState, CLFConfig, LyapunovConfig
from .nn import init_policy_params, policy_apply
from .lyapunov import init_lyapunov_params, lyapunov_matrix, lyapunov_value_and_grad
from .shield import clf_shield
from .adaptive import init_adaptive_state, adaptive_update_simple
from .integrator import rk4_step_generic
from .systems import get_system, list_systems


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(theta: Array) -> Array:
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def _sanitize_grads(grads):
    return jax.tree.map(
        lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
    )


# ---------------------------------------------------------------------------
# Cost (generic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostConfig:
    state_weights: Tuple[float, ...]
    u_weight: float = 0.01
    terminal_scale: float = 10.0
    proj_weight: float = 0.1
    infeasible_weight: float = 50.0


def _stage_cost(x: Array, u: Array, u_nom: Array, feasible: Array,
                cfg: CostConfig, wrap_state_fn) -> Array:
    """Quadratic stage cost with optional angle wrapping."""
    x_wrapped = wrap_state_fn(x)
    w = jnp.array(cfg.state_weights)
    state_cost = jnp.sum(w * x_wrapped ** 2)
    u_cost = cfg.u_weight * jnp.sum(u ** 2)
    proj_cost = cfg.proj_weight * jnp.sum((u - u_nom) ** 2)
    infeas_cost = cfg.infeasible_weight * (1.0 - feasible)
    return state_cost + u_cost + proj_cost + infeas_cost


def _terminal_cost(xT: Array, cfg: CostConfig, wrap_state_fn) -> Array:
    xT_wrapped = wrap_state_fn(xT)
    w = jnp.array(cfg.state_weights)
    return cfg.terminal_scale * jnp.sum(w * xT_wrapped ** 2)


# ---------------------------------------------------------------------------
# Policy modes
# ---------------------------------------------------------------------------

def _nn_policy(policy_params, x, spec, hidden_sizes):
    """Pure NN policy."""
    obs = spec["make_obs"](x)
    ctrl_dim = spec["ctrl_dim"]
    if ctrl_dim == 1:
        p = spec["params"]
        return policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes, out_dim=1)
    else:
        u_min = spec["u_min"]
        u_max = spec["u_max"]
        return policy_apply(policy_params, obs, u_min, u_max, hidden_sizes, out_dim=ctrl_dim)


def _hybrid_policy(policy_params, x, lqr_K, spec, hidden_sizes):
    """Blend LQR (near equilibrium) and NN (far from equilibrium).

    Blending is based on the Lyapunov-like distance from equilibrium.
    For systems with angles, uses cos(angle) for smooth blending.
    """
    u_nn = _nn_policy(policy_params, x, spec, hidden_sizes)
    p = spec["params"]

    if spec["ctrl_dim"] == 1:
        u_lqr = jnp.clip((-lqr_K @ x).squeeze(), p.u_min, p.u_max)
    else:
        u_eq = spec.get("u_eq", jnp.zeros(spec["ctrl_dim"]))
        u_lqr = u_eq - (lqr_K @ x)
        u_min = spec["u_min"]
        u_max = spec["u_max"]
        u_lqr = jnp.clip(u_lqr, u_min, u_max)

    # Blending weight: 1.0 near equilibrium (use LQR), 0.0 far away (use NN)
    x_eq = jnp.array(spec["x_eq"])
    err = jnp.linalg.norm(x - x_eq)
    alpha = jax.nn.sigmoid(8.0 * (1.0 - err))

    return alpha * u_lqr + (1.0 - alpha) * u_nn


# ---------------------------------------------------------------------------
# Control bounds helper
# ---------------------------------------------------------------------------

def _get_u_bounds(spec):
    """Return (u_min, u_max) as arrays."""
    if spec["ctrl_dim"] == 1:
        p = spec["params"]
        return jnp.array(p.u_min), jnp.array(p.u_max)
    else:
        return spec["u_min"], spec["u_max"]


def _clip_u(u, spec):
    u_min, u_max = _get_u_bounds(spec)
    return jnp.clip(u, u_min, u_max)


# ---------------------------------------------------------------------------
# Episode rollout (generic)
# ---------------------------------------------------------------------------

def _episode_rollout(
    policy_params: Dict[str, Any],
    lyap_params: Dict[str, Any] | None,
    x0: Array,
    a_true: Array,
    spec: Dict[str, Any],
    hidden_sizes: Tuple[int, ...],
    lqr_K: Array,
    horizon: int,
    dt: float,
    cost_cfg: CostConfig,
    clf_cfg: CLFConfig | None,
    lyap_cfg: LyapunovConfig | None,
    adapt_cfg: AdaptiveConfig | None,
    policy_mode: str,
) -> Tuple[Array, Dict[str, Array]]:
    p = spec["params"]
    x_max = 20.0
    use_clf = (policy_mode == "clf" and clf_cfg is not None
               and clf_cfg.enabled and lyap_params is not None)
    use_adapt = (adapt_cfg is not None and adapt_cfg.adapt_enabled)
    wrap_fn = spec["wrap_state"]
    affine_fn = spec["affine_terms_fn"]
    dynamics_fn = spec["dynamics_fn"]

    if use_adapt:
        adaptive_state0 = init_adaptive_state(p, adapt_cfg)
    else:
        adaptive_state0 = AdaptiveState(
            a_hat=jnp.array(0.0), info=jnp.array(1e-6), radius=jnp.array(0.0))

    def body(carry, _):
        x_raw, adaptive_state = carry
        x = jnp.clip(x_raw, -x_max, x_max)
        x = wrap_fn(x)

        if policy_mode == "hybrid":
            u_nom = _hybrid_policy(policy_params, x, lqr_K, spec, hidden_sizes)
        else:
            u_nom = _nn_policy(policy_params, x, spec, hidden_sizes)

        if use_clf:
            u_shield, shield_aux = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=adaptive_state,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_terms_fn=affine_fn,
            )
            u = _clip_u(u_shield, spec)
            feasible = shield_aux.get("feasible", jnp.array(1.0))
        else:
            u = u_nom
            feasible = jnp.array(1.0)

        if use_adapt:
            adaptive_next = adaptive_update_simple(
                adaptive_state, x, u, a_true, dt, p, adapt_cfg,
                affine_terms_fn=affine_fn, dynamics_fn=dynamics_fn,
            )
        else:
            adaptive_next = adaptive_state

        x_next = rk4_step_generic(dynamics_fn, x, u, a_true, dt, p)
        cost = _stage_cost(x, u, u_nom, feasible, cost_cfg, wrap_fn)
        return (x_next, adaptive_next), (cost, feasible)

    (xT, _), (costs, feasibles) = jax.lax.scan(
        body, (x0, adaptive_state0), jnp.arange(horizon))
    xT = wrap_fn(xT)

    mean_stage = jnp.mean(costs)
    terminal = _terminal_cost(xT, cost_cfg, wrap_fn)
    total_loss = jnp.nan_to_num(mean_stage + terminal, nan=1e4)

    metrics = {
        "loss": total_loss,
        "terminal_norm": jnp.linalg.norm(xT),
        "mean_stage_cost": mean_stage,
        "terminal_cost": terminal,
        "mean_feasible": jnp.mean(feasibles),
    }
    return total_loss, metrics


def _batched_loss(policy_params, lyap_params, batch_x0, batch_a,
                  spec, hidden_sizes, lqr_K, horizon, dt,
                  cost_cfg, clf_cfg, lyap_cfg, adapt_cfg, policy_mode):
    def single(x0, a_true):
        return _episode_rollout(
            policy_params, lyap_params, x0, a_true,
            spec, hidden_sizes, lqr_K, horizon, dt,
            cost_cfg, clf_cfg, lyap_cfg, adapt_cfg, policy_mode,
        )
    losses, metrics = jax.vmap(single)(batch_x0, batch_a)
    mean_loss = jnp.mean(losses)
    mean_metrics = jax.tree.map(jnp.mean, metrics)
    return mean_loss, mean_metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _default_run_dir(system: str, policy_mode: str, epochs: int, horizon: int,
                     pool_size: int, batch_size: int, lr: float) -> str:
    ts = datetime.now().strftime("%m%d_%H%M")
    return (f"runs/{system}_{policy_mode}_{ts}_ep{epochs}_h{horizon}"
            f"_pool{pool_size}_bs{batch_size}_lr{lr}")


def train(
    system: str = "cartpole",
    policy_mode: str | None = None,
    # Training
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    pool_size: int = 2048,
    horizon: int = 200,
    dt: float = 0.02,
    max_grad_norm: float = 10.0,
    # CLF
    clf_enabled: bool = False,
    learn_lyap: bool = False,
    lambda_clf: float = 0.5,
    lyap_mode: str = "quadratic_fixed",
    # Sampling
    region_scale: float = 1.0,
    a_true: float = 0.0,
    # Saving
    save_dir: str | None = None,
    seed: int = 0,
    log_every: int = 5,
    # Cost overrides (optional)
    cost_overrides: Dict[str, float] | None = None,
) -> Tuple[Dict[str, Any], list]:
    """Train a controller for the specified system.

    Returns (save_data, history).
    """
    if learn_lyap:
        clf_enabled = True
    if policy_mode is None:
        policy_mode = "clf" if clf_enabled else "nn_only"

    spec = get_system(system)
    p = spec["params"]
    state_dim = spec["state_dim"]
    ctrl_dim = spec["ctrl_dim"]
    obs_dim = spec["obs_dim"]
    hidden_sizes = (64, 64)

    # LQR
    P_lqr, lqr_K = spec["solve_lqr"](p, spec["default_lqr_Q"], spec["default_lqr_R"])
    print(f"LQR gain K shape: {lqr_K.shape}")

    # Cost config
    cost_w = dict(spec["default_cost_weights"])
    if clf_enabled:
        cost_w["proj_weight"] = 0.1
        cost_w["infeasible_weight"] = 50.0
    if cost_overrides:
        cost_w.update(cost_overrides)
    cost_cfg = CostConfig(
        state_weights=tuple(cost_w["state_weights"]),
        u_weight=cost_w["u_weight"],
        terminal_scale=cost_w["terminal_scale"],
        proj_weight=cost_w.get("proj_weight", 0.1),
        infeasible_weight=cost_w.get("infeasible_weight", 50.0),
    )

    # Save directory
    if save_dir is None:
        save_dir = _default_run_dir(system, policy_mode, epochs, horizon,
                                     pool_size, batch_size, lr)
    os.makedirs(save_dir, exist_ok=True)

    # Policy init
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    policy_params = init_policy_params(init_key, obs_dim, hidden_sizes, out_dim=ctrl_dim)

    # CLF / Lyapunov
    clf_cfg = CLFConfig(enabled=clf_enabled, lambda_clf=lambda_clf)
    adapt_cfg = AdaptiveConfig(adapt_enabled=False)

    if clf_enabled:
        key, lyap_key = jax.random.split(key)
        if learn_lyap:
            lyap_mode_str = "quadratic_learned"
        else:
            lyap_mode_str = lyap_mode
        lyap_cfg = LyapunovConfig(
            mode=lyap_mode_str,
            state_dim=state_dim,
            x_eq=spec["x_eq"],
            P_init=P_lqr,
        )
        lyap_params = init_lyapunov_params(lyap_key, lyap_cfg)
    else:
        lyap_cfg = None
        lyap_params = None

    # Optimizer
    steps_per_epoch = pool_size // batch_size
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )

    # Build trainable params
    if learn_lyap:
        lyap_static = {k: v for k, v in lyap_params.items() if not hasattr(v, 'shape')}
        lyap_trainable = {k: v for k, v in lyap_params.items() if hasattr(v, 'shape')}
        all_params = {"policy": policy_params, "lyap": lyap_trainable}
    else:
        all_params = policy_params

    opt_state = optimizer.init(all_params)

    # Step function
    if learn_lyap:
        @jax.jit
        def step_fn(all_params, opt_state, batch_x0, batch_a):
            def loss_fn(params):
                full_lyap = {**lyap_static, **params["lyap"]}
                return _batched_loss(
                    params["policy"], full_lyap, batch_x0, batch_a,
                    spec, hidden_sizes, lqr_K, horizon, dt,
                    cost_cfg, clf_cfg, lyap_cfg, adapt_cfg, policy_mode,
                )
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(all_params)
            grad_norm = optax.global_norm(grads)
            grads = _sanitize_grads(grads)
            updates, new_opt_state = optimizer.update(grads, opt_state, all_params)
            new_params = optax.apply_updates(all_params, updates)
            metrics["grad_norm"] = grad_norm
            return new_params, new_opt_state, loss, metrics
    else:
        @jax.jit
        def step_fn(params, opt_state, batch_x0, batch_a):
            def loss_fn(pp):
                return _batched_loss(
                    pp, lyap_params, batch_x0, batch_a,
                    spec, hidden_sizes, lqr_K, horizon, dt,
                    cost_cfg, clf_cfg, lyap_cfg, adapt_cfg, policy_mode,
                )
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            grad_norm = optax.global_norm(grads)
            grads = _sanitize_grads(grads)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            metrics["grad_norm"] = grad_norm
            return new_params, new_opt_state, loss, metrics

    # Print config
    print(f"\n{'='*60}")
    print(f"  {system} | mode={policy_mode}")
    print(f"  {epochs} epochs, horizon={horizon}, dt={dt}")
    print(f"  pool={pool_size}, batch={batch_size}, "
          f"{steps_per_epoch} steps/epoch")
    print(f"  lr={lr}, grad_clip={max_grad_norm}")
    if clf_enabled:
        lyap_str = "learned P" if learn_lyap else lyap_mode
        print(f"  CLF: ON (lambda={lambda_clf}, {lyap_str})")
    else:
        print(f"  CLF: OFF")
    print(f"  Region scale: {region_scale}")
    print(f"  Saving to: {save_dir}")
    print(f"{'='*60}\n")

    # Training loop
    history = []
    best_loss = float("inf")
    best_params = all_params
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        key, pool_key = jax.random.split(key)
        pool_x0, pool_a = spec["sample_ics"](pool_key, pool_size, region_scale, a_true)

        epoch_loss = 0.0
        epoch_grad = 0.0
        epoch_term = 0.0
        epoch_feas = 0.0

        for step_i in range(steps_per_epoch):
            idx = step_i * batch_size
            batch_x0 = pool_x0[idx:idx + batch_size]
            batch_a = pool_a[idx:idx + batch_size]

            all_params, opt_state, loss, metrics = step_fn(
                all_params, opt_state, batch_x0, batch_a)

            epoch_loss += float(loss)
            epoch_grad += float(metrics.get("grad_norm", 0.0))
            epoch_term += float(metrics.get("terminal_norm", 0.0))
            epoch_feas += float(metrics.get("mean_feasible", 1.0))

            global_step = (epoch - 1) * steps_per_epoch + step_i + 1
            record = {"step": global_step, "epoch": epoch}
            for k, v in metrics.items():
                record[k] = float(v)
            history.append(record)

        n = steps_per_epoch
        avg_loss = epoch_loss / n
        avg_grad = epoch_grad / n
        avg_term = epoch_term / n
        avg_feas = epoch_feas / n

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = jax.tree.map(lambda x: x.copy(), all_params)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            elapsed = time.time() - t0
            feas_str = f"feas {avg_feas:.3f} | " if clf_enabled else ""
            print(
                f"epoch {epoch:4d}/{epochs} | "
                f"loss {avg_loss:9.4f} | "
                f"term {avg_term:7.3f} | "
                f"{feas_str}"
                f"grad {avg_grad:8.4f} | "
                f"best {best_loss:8.4f} | "
                f"{elapsed:.1f}s"
            )

    # Extract final params
    save_all = best_params if best_loss < float("inf") else all_params
    if learn_lyap:
        policy_final = save_all["policy"]
        lyap_final = {**lyap_static, **save_all["lyap"]}
    else:
        policy_final = save_all
        lyap_final = lyap_params

    # Save
    save_data = {
        "system": system,
        "nn": jax.device_get(policy_final),
        "lqr_K": jax.device_get(lqr_K),
        "policy_mode": policy_mode,
        "clf_enabled": clf_enabled,
        "ctrl_dim": ctrl_dim,
        "obs_dim": obs_dim,
        "hidden_sizes": hidden_sizes,
    }
    if lyap_final is not None:
        save_data["lyap_params"] = jax.device_get(lyap_final)
    if clf_enabled:
        save_data["P_lqr"] = jax.device_get(P_lqr)
        save_data["lambda_clf"] = lambda_clf

    params_path = os.path.join(save_dir, "policy_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nSaved policy -> {params_path}")

    hist_path = os.path.join(save_dir, "history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    run_config = {
        "system": system, "policy_mode": policy_mode,
        "clf_enabled": clf_enabled, "learn_lyap": learn_lyap,
        "lambda_clf": lambda_clf, "region_scale": region_scale,
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "pool_size": pool_size, "horizon": horizon, "dt": dt,
        "max_grad_norm": max_grad_norm, "seed": seed,
        "lyap_mode": lyap_mode, "a_true": a_true,
    }
    config_path = os.path.join(save_dir, "run_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(run_config, f)

    _plot_training(history, save_dir, clf_enabled)

    if learn_lyap and lyap_final is not None:
        P_final = lyapunov_matrix(lyap_final, lyap_cfg)
        if P_final is not None:
            eigs = jnp.linalg.eigvalsh(P_final)
            print(f"Learned P eigenvalues: {eigs}")

    return save_data, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history: list, save_dir: str, clf_enabled: bool) -> None:
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]

    ncols = 3 if clf_enabled else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    axes = axes.flat

    axes[0].plot(steps, losses)
    axes[0].set(xlabel="step", ylabel="loss", title="Training loss")
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].grid(True, alpha=0.3)

    norms = [h.get("terminal_norm", 0) for h in history]
    axes[1].plot(steps, norms, color="tab:orange")
    axes[1].set(xlabel="step", ylabel="|x_T|", title="Terminal norm")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified training for control-affine systems")
    parser.add_argument("--system", type=str, default="cartpole",
                        choices=list_systems(),
                        help="System to train on")
    parser.add_argument("--policy-mode", type=str, default=None,
                        choices=["nn_only", "hybrid", "clf"],
                        help="Policy mode (default: clf if --clf else nn_only)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--clf", action="store_true", help="Enable CLF shield")
    parser.add_argument("--learn-lyap", action="store_true",
                        help="Jointly learn Lyapunov P (implies --clf)")
    parser.add_argument("--lyap-mode", type=str, default="quadratic_fixed",
                        choices=["quadratic_fixed", "quadratic_learned", "mlp_psd"])
    parser.add_argument("--lambda-clf", type=float, default=0.5)
    parser.add_argument("--region-scale", type=float, default=1.0)
    parser.add_argument("--a-true", type=float, default=0.0,
                        help="True uncertainty parameter value")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    train(
        system=args.system,
        policy_mode=args.policy_mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        horizon=args.horizon,
        dt=args.dt,
        max_grad_norm=args.max_grad_norm,
        clf_enabled=args.clf,
        learn_lyap=args.learn_lyap,
        lambda_clf=args.lambda_clf,
        lyap_mode=args.lyap_mode,
        region_scale=args.region_scale,
        a_true=args.a_true,
        save_dir=args.save_dir,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
