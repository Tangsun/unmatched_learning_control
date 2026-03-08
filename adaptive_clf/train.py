"""Training loop: learn a swing-up policy via differentiable simulation."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from .configs import AcrobotParams, AdaptiveConfig, CLFConfig, LyapunovConfig, RolloutConfig
from .acrobot import solve_lqr_P
from .lyapunov import init_lyapunov_params, lyapunov_matrix
from .rollout import (
    batched_rollout_loss,
    default_experiment_setup,
    sample_curriculum_batch,
    sample_mixed_batch,
    sample_uniform_batch,
    value_and_grad_loss,
    value_and_grad_joint,
)


def _sanitize_grads(grads):
    """Replace NaN/inf gradients with zero so the optimizer doesn't diverge."""
    return jax.tree.map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
        grads,
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
    sampling: str = "uniform",
    pool_size: int = 4096,
    clf_enabled: bool = False,
    learn_lyap: bool = False,
    region_scale: float = 1.0,
    lqr_blend: bool = False,
    lqr_V_threshold: float = 5.0,
    lqr_temperature: float = 1.0,
    terminal_cost_scale: float = 0.0,
    Q_stage_scale: float = 1.0,
    u_clip: float = 100.0,
    n_substeps: int = 1,
    bptt_window: int = 0,
    phases: List[Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], list]:
    """Train a swing-up policy and return (policy_params, loss_history).

    If *phases* is provided, run multi-phase curriculum training.  Each phase
    dict may override: n_steps, lr, sampling, region_scale, terminal_cost_scale,
    Q_stage_scale, lqr_V_threshold, lqr_temperature.
    """
    if learn_lyap:
        clf_enabled = True
    if save_dir is None:
        save_dir = _default_run_dir(n_steps, lr, batch_size, horizon)
    if warmup_steps is None:
        warmup_steps = n_steps // 2
    key = jax.random.PRNGKey(seed)
    p = AcrobotParams()

    Q_lqr = jnp.diag(jnp.array([40.0, 40.0, 8.0, 8.0]))
    R_lqr = jnp.array([[0.5]])
    P_lqr, K_lqr = solve_lqr_P(p, Q=Q_lqr, R=R_lqr, a_nom=0.0)
    print(f"P_lqr finite: {bool(jnp.all(jnp.isfinite(P_lqr)))}")
    print(f"K_lqr = {K_lqr}")

    K_lqr_arr = jnp.asarray(K_lqr) if lqr_blend else None
    P_lqr_arr = jnp.asarray(P_lqr) if lqr_blend else None

    hidden_sizes = (64, 64)
    key, setup_key = jax.random.split(key)
    adapt_cfg = AdaptiveConfig(adapt_enabled=False)

    if learn_lyap:
        from .nn import init_policy_params as _init_pol
        k_policy, k_lyap = jax.random.split(setup_key)
        obs_dim = 4
        policy_params = _init_pol(k_policy, obs_dim=obs_dim, hidden_sizes=hidden_sizes)
        lyap_cfg = LyapunovConfig(
            mode="quadratic_learned",
            state_dim=4,
            x_eq=(0.0, 0.0, 0.0, 0.0),
            P_init=P_lqr,
        )
        lyap_params = init_lyapunov_params(k_lyap, lyap_cfg)
    else:
        policy_params, lyap_params, lyap_cfg = default_experiment_setup(
            key=setup_key, p=p, P_lqr=P_lqr, policy_hidden_sizes=hidden_sizes,
            adapt_cfg=adapt_cfg,
        )
    clf_cfg = CLFConfig(enabled=clf_enabled)

    os.makedirs(save_dir, exist_ok=True)
    history: list = []

    if phases is not None:
        opt_var = policy_params
        for phase_idx, phase_cfg in enumerate(phases):
            print(f"\n{'#'*60}")
            print(f"  PHASE {phase_idx + 1}/{len(phases)}")
            print(f"  config: {phase_cfg}")
            print(f"{'#'*60}")

            ph_n = phase_cfg.get("n_steps", n_steps)
            ph_lr = phase_cfg.get("lr", lr)
            ph_sampling = phase_cfg.get("sampling", sampling)
            ph_region = phase_cfg.get("region_scale", region_scale)
            ph_terminal = phase_cfg.get("terminal_cost_scale", terminal_cost_scale)
            ph_Q_scale = phase_cfg.get("Q_stage_scale", Q_stage_scale)
            ph_V_thresh = phase_cfg.get("lqr_V_threshold", lqr_V_threshold)
            ph_V_temp = phase_cfg.get("lqr_temperature", lqr_temperature)
            ph_warmup = phase_cfg.get("warmup_steps", ph_n // 2)
            ph_u_clip = phase_cfg.get("u_clip", u_clip)
            ph_horizon = phase_cfg.get("horizon", horizon)
            ph_bptt = phase_cfg.get("bptt_window", bptt_window)
            ph_R_energy = phase_cfg.get("R_energy", 0.0)

            Q_track_scaled = tuple(
                tuple(ph_Q_scale * v for v in row)
                for row in RolloutConfig().Q_track
            )
            rollout_cfg = RolloutConfig(
                horizon=ph_horizon, dt=0.02,
                Q_track=Q_track_scaled,
                Q_terminal_scale=ph_terminal,
                R_energy=ph_R_energy,
                u_clip=ph_u_clip,
                n_substeps=n_substeps,
                bptt_window=ph_bptt,
                lqr_blend=lqr_blend,
                lqr_V_threshold=ph_V_thresh,
                lqr_temperature=ph_V_temp,
            )

            optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(ph_lr),
            )
            opt_state = optimizer.init(opt_var)

            dq1_range = float(ph_region * jnp.pi)
            dq2_range = float(ph_region * jnp.pi)
            w_range = float(ph_region * 2.0)
            use_curriculum = (ph_sampling == "curriculum")
            resample_every = max(ph_n // 10, 1) if use_curriculum else 0

            @jax.jit
            def step_fn(pp, os_, bx, ba):
                (loss, metrics), grads = value_and_grad_loss(
                    policy_params=pp, lyap_params=lyap_params,
                    batch_x0=bx, batch_a_true=ba,
                    p=p, policy_hidden_sizes=hidden_sizes,
                    lyap_cfg=lyap_cfg, adapt_cfg=adapt_cfg,
                    clf_cfg=clf_cfg, rollout_cfg=rollout_cfg,
                    K_lqr=K_lqr_arr, P_lqr=P_lqr_arr,
                )
                gn = optax.global_norm(grads)
                grads = _sanitize_grads(grads)
                upd, new_os = optimizer.update(grads, os_, pp)
                new_pp = optax.apply_updates(pp, upd)
                metrics["grad_norm"] = gn
                return new_pp, new_os, loss, metrics

            use_mixed = (ph_sampling == "mixed")
            batch_x0 = batch_a = None
            t0 = time.time()
            for step in range(1, ph_n + 1):
                if use_mixed:
                    key, bk = jax.random.split(key)
                    batch_x0, batch_a = sample_mixed_batch(
                        bk, batch_size, p, region_scale=ph_region)
                elif use_curriculum:
                    progress = min(step / max(ph_warmup, 1), 1.0)
                    need = (batch_x0 is None or not fixed_batch
                            or (fixed_batch and step % resample_every == 1))
                    if need:
                        key, bk = jax.random.split(key)
                        batch_x0, batch_a = sample_curriculum_batch(
                            bk, batch_size, p, progress=progress,
                            region_scale=ph_region,
                        )
                else:
                    need = (batch_x0 is None or not fixed_batch)
                    if need:
                        key, bk = jax.random.split(key)
                        batch_x0, batch_a = sample_uniform_batch(
                            bk, batch_size, p,
                            dq1_range=dq1_range, dq2_range=dq2_range,
                            w_range=w_range,
                        )

                opt_var, opt_state, loss, metrics = step_fn(
                    opt_var, opt_state, batch_x0, batch_a)

                loss_val = float(loss)
                global_step = sum(ph.get("n_steps", n_steps) for ph in phases[:phase_idx]) + step
                record = {"step": global_step, "phase": phase_idx + 1,
                          "loss": loss_val}
                for k, v in metrics.items():
                    record[k] = float(v)
                history.append(record)

                if step == 1 or step % log_every == 0 or step == ph_n:
                    elapsed = time.time() - t0
                    print(
                        f"  ph{phase_idx+1} step {step:5d}/{ph_n} | "
                        f"loss {loss_val:10.2f} | "
                        f"term_norm {record['terminal_norm']:7.3f} | "
                        f"grad {record['grad_norm']:8.4f} | "
                        f"{elapsed:.1f}s"
                    )

        policy_params = opt_var

    else:
        Q_track_scaled = tuple(
            tuple(Q_stage_scale * v for v in row)
            for row in RolloutConfig().Q_track
        )
        rollout_cfg = RolloutConfig(
            horizon=horizon, dt=0.02,
            Q_track=Q_track_scaled,
            Q_terminal_scale=terminal_cost_scale,
            u_clip=u_clip,
            n_substeps=n_substeps,
            bptt_window=bptt_window,
            lqr_blend=lqr_blend,
            lqr_V_threshold=lqr_V_threshold,
            lqr_temperature=lqr_temperature,
        )

        dq1_range = float(region_scale * jnp.pi)
        dq2_range = float(region_scale * jnp.pi)
        w_range = float(region_scale * 2.0)

        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr),
        )

        if learn_lyap:
            lyap_static = {k: v for k, v in lyap_params.items()
                           if not hasattr(v, 'shape')}
            lyap_trainable = {k: v for k, v in lyap_params.items()
                             if hasattr(v, 'shape')}
            all_params = {"policy": policy_params, "lyap": lyap_trainable}
            opt_state = optimizer.init(all_params)

            @jax.jit
            def step_fn(all_params, opt_state, batch_x0, batch_a):
                def loss_fn(params):
                    full_lyap = {**lyap_static, **params["lyap"]}
                    return batched_rollout_loss(
                        policy_params=params["policy"],
                        lyap_params=full_lyap,
                        batch_x0=batch_x0,
                        batch_a_true=batch_a,
                        p=p,
                        policy_hidden_sizes=hidden_sizes,
                        lyap_cfg=lyap_cfg,
                        adapt_cfg=adapt_cfg,
                        clf_cfg=clf_cfg,
                        rollout_cfg=rollout_cfg,
                        K_lqr=K_lqr_arr,
                        P_lqr=P_lqr_arr,
                    )
                (loss, metrics), grads = jax.value_and_grad(
                    loss_fn, has_aux=True)(all_params)
                grad_norm = optax.global_norm(grads)
                grads = _sanitize_grads(grads)
                updates, new_opt_state = optimizer.update(grads, opt_state, all_params)
                new_params = optax.apply_updates(all_params, updates)
                metrics["grad_norm"] = grad_norm
                return new_params, new_opt_state, loss, metrics
        else:
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
                    K_lqr=K_lqr_arr,
                    P_lqr=P_lqr_arr,
                )
                grad_norm = optax.global_norm(grads)
                grads = _sanitize_grads(grads)
                updates, new_opt_state = optimizer.update(grads, opt_state, policy_params)
                new_params = optax.apply_updates(policy_params, updates)
                metrics["grad_norm"] = grad_norm
                return new_params, new_opt_state, loss, metrics

        use_curriculum = (sampling == "curriculum")
        use_minibatch = (sampling == "minibatch")
        resample_every = max(n_steps // 10, 1) if use_curriculum else 0

        if use_minibatch:
            steps_per_epoch = pool_size // batch_size
            batch_desc = (f"minibatch (pool={pool_size}, mb={batch_size}, "
                          f"{steps_per_epoch} steps/epoch)")
        elif use_curriculum:
            batch_desc = f"curriculum (warmup={warmup_steps}, resample every {resample_every})"
        elif fixed_batch:
            batch_desc = "uniform fixed"
        else:
            batch_desc = "uniform random"

        print(f"\n{'='*60}")
        print(f"  Swing-up training: {n_steps} steps, horizon={horizon}, "
              f"batch={batch_size}, lr={lr}")
        print(f"  Sampling: {batch_desc}")
        if learn_lyap:
            shield_str = "CLF + learned P (init from LQR)"
        elif clf_enabled:
            shield_str = "CLF ENABLED (fixed LQR Lyapunov)"
        else:
            shield_str = "DISABLED"
        print(f"  Shield: {shield_str} | Adapt: OFF | Friction: a=0")
        lqr_str = (f"ON (V_thresh={lqr_V_threshold:.1f}, temp={lqr_temperature:.1f})"
                    if lqr_blend else "OFF")
        print(f"  LQR blend: {lqr_str}")
        print(f"  Terminal cost scale: {terminal_cost_scale:.1f} | "
              f"Q stage scale: {Q_stage_scale:.2f} | u_clip: {u_clip:.0f}")
        print(f"  Region: scale={region_scale} "
              f"(dq1,dq2,w in +/-[{dq1_range:.2f},{dq2_range:.2f},{w_range:.2f}])")
        print(f"  Saving to: {save_dir}")
        print(f"{'='*60}\n")

        opt_var = all_params if learn_lyap else policy_params

        pool_x0 = pool_a = None
        if use_minibatch:
            key, pool_key = jax.random.split(key)
            pool_x0, pool_a = sample_uniform_batch(
                pool_key, pool_size, p,
                dq1_range=dq1_range, dq2_range=dq2_range, w_range=w_range,
            )
            print(f"Sampled pool of {pool_size} ICs (region_scale={region_scale})")

        batch_x0 = batch_a = None
        t0 = time.time()
        for step in range(1, n_steps + 1):
            if use_minibatch:
                idx_in_epoch = (step - 1) % steps_per_epoch
                epoch = (step - 1) // steps_per_epoch + 1

                if idx_in_epoch == 0:
                    key, shuffle_key = jax.random.split(key)
                    perm = jax.random.permutation(shuffle_key, pool_size)
                    pool_x0 = pool_x0[perm]
                    pool_a = pool_a[perm]

                start = idx_in_epoch * batch_size
                batch_x0 = pool_x0[start : start + batch_size]
                batch_a = pool_a[start : start + batch_size]

            elif use_curriculum:
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
                        region_scale=region_scale,
                    )
            else:
                need_resample = (batch_x0 is None or not fixed_batch)
                if need_resample:
                    key, batch_key = jax.random.split(key)
                    batch_x0, batch_a = sample_uniform_batch(
                        batch_key, batch_size, p,
                        dq1_range=dq1_range, dq2_range=dq2_range, w_range=w_range,
                    )

            opt_var, opt_state, loss, metrics = step_fn(
                opt_var, opt_state, batch_x0, batch_a,
            )

            loss_val = float(loss)
            record = {"step": step, "loss": loss_val}
            if use_minibatch:
                record["epoch"] = epoch
            for k, v in metrics.items():
                record[k] = float(v)
            history.append(record)

            if step == 1 or step % log_every == 0 or step == n_steps:
                elapsed = time.time() - t0
                epoch_str = f" ep {epoch}" if use_minibatch else ""
                lyap_str = ""
                if learn_lyap:
                    P_cur = lyapunov_matrix({**lyap_static, **opt_var["lyap"]}, lyap_cfg)
                    eigs = jnp.linalg.eigvalsh(P_cur)
                    lyap_str = f" | P_eig [{float(eigs[0]):.2f}, {float(eigs[-1]):.2f}]"
                print(
                    f"step {step:5d}/{n_steps}{epoch_str} | "
                    f"loss {loss_val:10.2f} | "
                    f"term_norm {record['terminal_norm']:7.3f} | "
                    f"grad {record['grad_norm']:8.4f}{lyap_str} | "
                    f"{elapsed:.1f}s"
                )

        if learn_lyap:
            policy_params = opt_var["policy"]
            lyap_params = {**lyap_static, **opt_var["lyap"]}
        else:
            policy_params = opt_var

    params_path = os.path.join(save_dir, "policy_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(jax.device_get(policy_params), f)
    print(f"\nSaved policy params to {params_path}")

    run_config = {
        "region_scale": region_scale,
        "dq1_range": float(region_scale * jnp.pi),
        "dq2_range": float(region_scale * jnp.pi),
        "w_range": float(region_scale * 2.0),
        "lqr_blend": lqr_blend,
        "lqr_V_threshold": lqr_V_threshold,
        "lqr_temperature": lqr_temperature,
        "clf_enabled": clf_enabled,
        "terminal_cost_scale": terminal_cost_scale,
        "Q_stage_scale": Q_stage_scale,
        "u_clip": u_clip,
    }
    if lqr_blend:
        run_config["K_lqr"] = jax.device_get(K_lqr_arr)
        run_config["P_lqr"] = jax.device_get(P_lqr_arr)
    config_path = os.path.join(save_dir, "run_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(run_config, f)

    if learn_lyap:
        lyap_path = os.path.join(save_dir, "lyap_params.pkl")
        with open(lyap_path, "wb") as f:
            pickle.dump(jax.device_get(lyap_params), f)
        P_final = lyapunov_matrix(lyap_params, lyap_cfg)
        eigs_final = jnp.linalg.eigvalsh(P_final)
        print(f"Saved lyap params to {lyap_path}")
        print(f"Learned P eigenvalues: {eigs_final}")

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

    if "phase" in history[0]:
        phase_boundaries = []
        for i in range(1, len(history)):
            if history[i].get("phase", 1) != history[i-1].get("phase", 1):
                phase_boundaries.append(history[i]["step"])
        for b in phase_boundaries:
            ax1.axvline(b, color="red", ls="--", alpha=0.5)
            ax2.axvline(b, color="red", ls="--", alpha=0.5)

    fig.tight_layout()
    out = os.path.join(save_dir, "training_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curve to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train acrobot swing-up policy")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--sampling", choices=["uniform", "curriculum", "minibatch"],
                        default="uniform")
    parser.add_argument("--pool-size", type=int, default=20000,
                        help="Pool size for minibatch sampling mode")
    parser.add_argument("--clf", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable CLF shield (fixed LQR Lyapunov)")
    parser.add_argument("--learn-lyap", action=argparse.BooleanOptionalAction, default=False,
                        help="Jointly learn quadratic Lyapunov P (implies --clf)")
    parser.add_argument("--region-scale", type=float, default=1.0,
                        help="Scale state-space region (1.0=full, 0.3=near upright)")

    parser.add_argument("--lqr-blend", action=argparse.BooleanOptionalAction, default=False,
                        help="Blend NN with LQR near upright")
    parser.add_argument("--lqr-V-threshold", type=float, default=5.0,
                        help="Lyapunov sublevel for LQR blend transition")
    parser.add_argument("--lqr-temperature", type=float, default=1.0,
                        help="Sharpness of LQR blend sigmoid")
    parser.add_argument("--terminal-cost", type=float, default=0.0,
                        help="Terminal cost scale (Q_terminal_scale)")
    parser.add_argument("--Q-stage-scale", type=float, default=1.0,
                        help="Multiply default Q_track by this factor")
    parser.add_argument("--u-clip", type=float, default=100.0,
                        help="Control clip in rollout (set large to disable)")
    parser.add_argument("--n-substeps", type=int, default=1,
                        help="RK4 integration substeps per control step")
    parser.add_argument("--bptt-window", type=int, default=0,
                        help="Truncated BPTT window (0=full backprop)")
    parser.add_argument("--phased", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable multi-phase curriculum training")

    args = parser.parse_args()

    phases = None
    if args.phased:
        total = args.n_steps
        p1 = max(total // 4, 100)
        p2 = max(total // 4, 100)
        p3 = total - p1 - p2
        phases = [
            {"n_steps": p1, "sampling": "uniform", "region_scale": 0.2,
             "terminal_cost_scale": 5.0, "Q_stage_scale": 1.0,
             "lr": args.lr},
            {"n_steps": p2, "sampling": "curriculum", "region_scale": 0.6,
             "terminal_cost_scale": 10.0, "Q_stage_scale": 0.5,
             "lr": args.lr},
            {"n_steps": p3, "sampling": "curriculum", "region_scale": 1.0,
             "terminal_cost_scale": 10.0, "Q_stage_scale": 0.3,
             "lr": args.lr * 0.5},
        ]

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
        sampling=args.sampling,
        pool_size=args.pool_size,
        clf_enabled=args.clf,
        learn_lyap=args.learn_lyap,
        region_scale=args.region_scale,
        lqr_blend=args.lqr_blend,
        lqr_V_threshold=args.lqr_V_threshold,
        lqr_temperature=args.lqr_temperature,
        terminal_cost_scale=args.terminal_cost,
        Q_stage_scale=args.Q_stage_scale,
        u_clip=args.u_clip,
        n_substeps=args.n_substeps,
        bptt_window=args.bptt_window,
        phases=phases,
    )


if __name__ == "__main__":
    main()
