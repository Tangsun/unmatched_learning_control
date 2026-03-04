"""Training loop: learn a swing-up policy via differentiable simulation."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from .configs import AcrobotParams, AdaptiveConfig, CLFConfig, RolloutConfig
from .acrobot import solve_lqr_P
from .rollout import (
    default_experiment_setup,
    sample_curriculum_batch,
    value_and_grad_loss,
)


def _default_run_dir(n_steps: int, lr: float, batch_size: int, horizon: int) -> str:
    ts = datetime.now().strftime("%m%d_%H%M")
    return f"runs/swingup_{ts}_s{n_steps}_h{horizon}_bs{batch_size}_lr{lr}"


def train_swingup(
    n_steps: int = 500,
    lr: float = 3e-4,
    batch_size: int = 32,
    horizon: int = 500,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
    save_dir: str | None = None,
    seed: int = 0,
    fixed_batch: bool = True,
    warmup_steps: int | None = None,
) -> Tuple[Dict[str, Any], list]:
    """Train a swing-up policy and return (policy_params, loss_history)."""
    if save_dir is None:
        save_dir = _default_run_dir(n_steps, lr, batch_size, horizon)
    if warmup_steps is None:
        warmup_steps = n_steps // 2
    key = jax.random.PRNGKey(seed)
    p = AcrobotParams()

    Q_lqr = jnp.diag(jnp.array([40.0, 40.0, 8.0, 8.0]))
    R_lqr = jnp.array([[0.5]])
    P_lqr, _ = solve_lqr_P(p, Q=Q_lqr, R=R_lqr, a_nom=0.0)
    print(f"P_lqr finite: {bool(jnp.all(jnp.isfinite(P_lqr)))}")

    hidden_sizes = (64, 64)
    key, setup_key = jax.random.split(key)
    policy_params, lyap_params, lyap_cfg = default_experiment_setup(
        key=setup_key, p=p, P_lqr=P_lqr, policy_hidden_sizes=hidden_sizes,
    )

    adapt_cfg = AdaptiveConfig()
    clf_cfg = CLFConfig(enabled=False)
    rollout_cfg = RolloutConfig(horizon=horizon, dt=0.02)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )
    opt_state = optimizer.init(policy_params)

    @jax.jit
    def step_fn(policy_params, opt_state, batch_x0, batch_a):
        (loss, metrics), grads = value_and_grad_loss(
            policy_params=policy_params,
            lyap_params=lyap_params,
            batch_x0=batch_x0,
            batch_a_true=batch_a,
            p=p,
            policy_hidden_sizes=hidden_sizes,
            lyap_cfg=lyap_cfg,
            adapt_cfg=adapt_cfg,
            clf_cfg=clf_cfg,
            rollout_cfg=rollout_cfg,
        )
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state, policy_params)
        new_params = optax.apply_updates(policy_params, updates)
        metrics["grad_norm"] = grad_norm
        return new_params, new_opt_state, loss, metrics

    os.makedirs(save_dir, exist_ok=True)
    history = []
    resample_every = max(n_steps // 10, 1)

    batch_mode = "fixed (resample every " + str(resample_every) + ")" if fixed_batch else "random"
    print(f"\n{'='*60}")
    print(f"  Swing-up training: {n_steps} steps, horizon={horizon}, "
          f"batch={batch_size}, lr={lr}")
    print(f"  Batch mode: {batch_mode} | Curriculum warmup: {warmup_steps} steps")
    print(f"  Shield: DISABLED | Friction: a=0")
    print(f"  Saving to: {save_dir}")
    print(f"{'='*60}\n")

    batch_x0 = batch_a = None
    t0 = time.time()
    for step in range(1, n_steps + 1):
        progress = min(step / max(warmup_steps, 1), 1.0)
        need_resample = (
            batch_x0 is None
            or not fixed_batch
            or (fixed_batch and step % resample_every == 1)
        )

        if need_resample:
            key, batch_key = jax.random.split(key)
            batch_x0, batch_a = sample_curriculum_batch(
                batch_key, batch_size, p, progress=progress,
            )

        policy_params, opt_state, loss, metrics = step_fn(
            policy_params, opt_state, batch_x0, batch_a,
        )

        loss_val = float(loss)
        record = {"step": step, "loss": loss_val, "progress": progress}
        for k, v in metrics.items():
            record[k] = float(v)
        history.append(record)

        if step == 1 or step % log_every == 0 or step == n_steps:
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{n_steps} | loss {loss_val:10.2f} | "
                f"term_norm {record['terminal_norm']:7.3f} | "
                f"grad_norm {record['grad_norm']:8.4f} | "
                f"prog {progress:.2f} | {elapsed:.1f}s"
            )

    params_path = os.path.join(save_dir, "policy_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(jax.device_get(policy_params), f)
    print(f"\nSaved policy params to {params_path}")

    hist_path = os.path.join(save_dir, "history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    _plot_training_curve(history, save_dir)

    return policy_params, history


def _plot_training_curve(history: list, save_dir: str) -> None:
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    term_norms = [h["terminal_norm"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, losses)
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("Training loss")
    ax1.set_yscale("symlog", linthresh=100)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, term_norms, color="tab:orange")
    ax2.set_xlabel("step")
    ax2.set_ylabel("|x_T|")
    ax2.set_title("Terminal state norm")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(save_dir, "training_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curve to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train acrobot swing-up policy")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-steps", type=int, default=None)
    args = parser.parse_args()

    train_swingup(
        n_steps=args.n_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        horizon=args.horizon,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        save_dir=args.save_dir,
        seed=args.seed,
        fixed_batch=args.fixed_batch,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
