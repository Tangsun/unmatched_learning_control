"""Joint policy + neural Lyapunov training with smooth CLF violation.

Train a policy and MLP-PSD Lyapunov function from scratch. Uses a smooth
CLF violation penalty (relu(Vdot + lambda*V)^2) instead of hard projection
during training. Region curriculum expands from a small neighborhood to
the full state space.

Usage:
    # Small region, no shield (V-fitting + policy)
    python -m adaptive_clf.train_lyapunov \
        --region-start 0.3 --region-end 0.3 --epochs 100 \
        --save-dir runs/lyap_small

    # Region curriculum: start small, expand
    python -m adaptive_clf.train_lyapunov \
        --region-start 0.3 --region-end 1.0 --epochs 200 \
        --save-dir runs/lyap_curriculum
"""

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

from .configs import Array, AdaptiveConfig, AdaptiveState, CLFConfig, LyapunovConfig
from .nn import init_policy_params
from .lyapunov import init_lyapunov_params, lyapunov_value_and_grad
from .shield import clf_shield
from .adaptive import adaptive_update_generic, adaptive_update_observer, init_adaptive_state
from .integrator import rk4_step_generic
from .systems import get_system, list_systems
from .train_unified import (
    _hybrid_policy, _nn_policy, _clip_u, _get_u_bounds, _sanitize_grads,
    CostConfig, _terminal_cost,
)


# ---------------------------------------------------------------------------
# Rollout with smooth Lyapunov violation
# ---------------------------------------------------------------------------

def _nn_policy_with_obs(policy_params, obs, spec, hidden_sizes):
    """NN policy from pre-built observation vector."""
    from .nn import policy_apply
    ctrl_dim = spec["ctrl_dim"]
    if ctrl_dim == 1:
        p = spec["params"]
        return policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes, out_dim=1)
    else:
        u_min = spec["u_min"]
        u_max = spec["u_max"]
        return policy_apply(policy_params, obs, u_min, u_max, hidden_sizes, out_dim=ctrl_dim)


def _episode_rollout_lyap(
    policy_params: Dict[str, Any],
    lyap_params: Dict[str, Any],
    x0: Array,
    a_true: Array,
    spec: Dict[str, Any],
    hidden_sizes: Tuple[int, ...],
    lqr_K: Array,
    horizon: int,
    dt: float,
    cost_cfg: CostConfig,
    lyap_cfg: LyapunovConfig,
    clf_cfg: CLFConfig,
    use_shield: bool,
    lambda_clf: float,
    w_violation: float,
    policy_mode: str,
    adapt_cfg: AdaptiveConfig | None = None,
    shield_diff: bool = False,
    alpha_max: float = 0.0,
    a_range: float = 0.0,
) -> Tuple[Array, Dict[str, Array]]:
    """Rollout with smooth CLF violation penalty."""
    p = spec["params"]
    x_max = 20.0
    wrap_fn = spec["wrap_state"]
    affine_fn = spec["affine_terms_fn"]
    dynamics_fn = spec["dynamics_fn"]
    make_obs = spec["make_obs"]
    use_adapt = adapt_cfg is not None and adapt_cfg.adapt_enabled
    state_dim = spec["state_dim"]
    use_observer = use_adapt and adapt_cfg.use_observer

    # Pass a_range so the initial radius matches the actual disturbance range
    adapt_state0 = (init_adaptive_state(p, adapt_cfg, state_dim=state_dim, x0=x0,
                                        a_range=a_range if a_range > 0 else None)
                    if use_adapt
                    else AdaptiveState(
                        a_hat=jnp.array(0.0), info=jnp.array(1e-6),
                        radius=jnp.array(0.0),
                        x_hat=jnp.zeros(state_dim),
                        w=jnp.zeros(state_dim),
                        eta=jnp.zeros(state_dim)))

    def body(carry, _):
        x_raw, adapt_st = carry
        x = jnp.clip(x_raw, -x_max, x_max)
        x = wrap_fn(x)

        # Build observation (optionally augmented with a_hat, radius)
        obs = make_obs(x)
        if use_adapt:
            a_hat_sg = jax.lax.stop_gradient(adapt_st.a_hat)
            rad_sg = jax.lax.stop_gradient(adapt_st.radius)
            obs = jnp.concatenate([obs, jnp.array([a_hat_sg, rad_sg])])

        # Nominal policy
        if policy_mode == "hybrid":
            u_nom = _hybrid_policy(policy_params, x, lqr_K, spec, hidden_sizes)
        else:
            u_nom = _nn_policy_with_obs(policy_params, obs, spec, hidden_sizes)

        # Optional CLF shield
        if use_shield:
            u_shield, shield_aux = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=adapt_st,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_terms_fn=affine_fn,
                alpha_max=alpha_max,
            )
            if shield_diff:
                # Differentiable shield: gradients flow through projection
                u = u_shield
            else:
                u = _clip_u(jax.lax.stop_gradient(u_shield), spec)
            feasible = shield_aux.get("feasible", jnp.array(1.0))
        else:
            u = _clip_u(u_nom, spec)
            feasible = jnp.array(1.0)

        # Smooth Vdot violation (computed on u_nom, not shielded u)
        V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
        xdot_nom = dynamics_fn(x, _clip_u(u_nom, spec), a_true, p)
        Vdot = gradV @ xdot_nom
        violation = jax.nn.relu(Vdot + lambda_clf * V) ** 2

        # Feasibility loss (skip when shield_diff with unconstrained u —
        # the halfspace is always feasible)
        if shield_diff:
            feas_penalty = jnp.array(0.0)
        else:
            f_x, g_x, y_x = affine_fn(x, p)
            LfV = gradV @ f_x
            LgV = gradV @ g_x
            b_scalar_feas = -lambda_clf * V - LfV
            u_lo, u_hi = _get_u_bounds(spec)
            best_per_channel = jnp.minimum(LgV * u_lo, LgV * u_hi)
            best_total = jnp.sum(best_per_channel)
            infeas_margin = jax.nn.relu(best_total - b_scalar_feas)
            feas_penalty = infeas_margin ** 2

        # State/control cost
        x_wrapped = wrap_fn(x)
        w = jnp.array(cost_cfg.state_weights)
        state_cost = jnp.sum(w * x_wrapped ** 2)
        u_cost = cost_cfg.u_weight * jnp.sum(u ** 2)
        proj_cost = cost_cfg.proj_weight * jnp.sum((u - u_nom) ** 2)

        step_cost = (state_cost + u_cost + proj_cost
                     + w_violation * violation
                     + w_violation * feas_penalty)

        # Adaptive update (stop_gradient so no backprop through estimator)
        if use_observer:
            adapt_next = adaptive_update_observer(
                adapt_st, x, u, dt, p, adapt_cfg,
                affine_terms_fn=affine_fn)
            adapt_next = jax.lax.stop_gradient(adapt_next)
        elif use_adapt:
            adapt_next = adaptive_update_generic(
                adapt_st, x, u, a_true, dt, p, adapt_cfg,
                affine_terms_fn=affine_fn, dynamics_fn=dynamics_fn)
            adapt_next = jax.lax.stop_gradient(adapt_next)
        else:
            adapt_next = adapt_st

        x_next = rk4_step_generic(dynamics_fn, x, u, a_true, dt, p)
        return (x_next, adapt_next), (step_cost, violation, V, feasible)

    (xT, _), (costs, violations, Vs, feasibles) = jax.lax.scan(
        body, (x0, adapt_state0), jnp.arange(horizon))
    xT = wrap_fn(xT)

    mean_stage = jnp.mean(costs)
    terminal = _terminal_cost(xT, cost_cfg, wrap_fn)
    total_loss = jnp.nan_to_num(mean_stage + terminal, nan=1e4)

    metrics = {
        "loss": total_loss,
        "terminal_norm": jnp.linalg.norm(xT),
        "mean_violation": jnp.mean(violations),
        "mean_V": jnp.mean(Vs),
        "mean_feasible": jnp.mean(feasibles),
    }
    return total_loss, metrics


def _batched_loss_lyap(
    policy_params, lyap_params, batch_x0, batch_a,
    spec, hidden_sizes, lqr_K, horizon, dt,
    cost_cfg, lyap_cfg, clf_cfg,
    use_shield, lambda_clf, w_violation, policy_mode,
    adapt_cfg=None, shield_diff=False, alpha_max=0.0,
    a_range=0.0,
):
    def single(x0, a_true):
        return _episode_rollout_lyap(
            policy_params, lyap_params, x0, a_true,
            spec, hidden_sizes, lqr_K, horizon, dt,
            cost_cfg, lyap_cfg, clf_cfg,
            use_shield, lambda_clf, w_violation, policy_mode,
            adapt_cfg=adapt_cfg, shield_diff=shield_diff,
            alpha_max=alpha_max, a_range=a_range,
        )
    losses, metrics = jax.vmap(single)(batch_x0, batch_a)
    mean_loss = jnp.mean(losses)
    mean_metrics = jax.tree.map(jnp.mean, metrics)
    return mean_loss, mean_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lyapunov(
    system: str = "cartpole",
    policy_mode: str = "nn_only",
    # Training
    epochs: int = 200,
    lr: float = 5e-4,
    batch_size: int = 64,
    pool_size: int = 2048,
    horizon: int = 200,
    dt: float = 0.02,
    max_grad_norm: float = 10.0,
    # Lyapunov
    lambda_clf: float = 0.1,
    w_violation_start: float = 50.0,
    w_violation_end: float = 5.0,
    lyap_hidden: Tuple[int, ...] = (64, 64),
    use_shield: bool = False,
    shield_diff: bool = False,
    # Region curriculum
    region_start: float = 0.3,
    region_end: float = 0.3,
    curriculum_frac: float = 0.5,
    # Sampling
    a_true: float = 0.0,
    a_range: float = 0.0,  # if >0, randomize a_true ~ U[-a_range, a_range]
    # Adaptive estimator
    use_adapt: bool = False,
    adapt_eta: float = 2e-2,
    # Observer-based adaptation
    use_observer: bool = False,
    observer_k: float = 5.0,
    observer_gamma: float = 5.0,
    # Shield projection gain cap (0 = no cap)
    alpha_max: float = 0.0,
    # Freeze Lyapunov (train policy only)
    freeze_lyap: bool = False,
    # Warm start
    warmstart_from: str | None = None,
    warmstart_lyap_only: bool = False,
    # Saving
    save_dir: str | None = None,
    seed: int = 0,
    log_every: int = 5,
) -> Tuple[Dict[str, Any], list]:
    """Train policy + neural Lyapunov function jointly from scratch."""

    spec = get_system(system)
    p = spec["params"]
    state_dim = spec["state_dim"]
    ctrl_dim = spec["ctrl_dim"]
    obs_dim = spec["obs_dim"]
    if use_adapt:
        obs_dim += 2  # augment with [a_hat, radius]
    hidden_sizes = (64, 64)

    # Adaptive config
    adapt_cfg = AdaptiveConfig(
        adapt_enabled=use_adapt, eta=adapt_eta,
        stopgrad_obs=True,
        use_observer=use_observer,
        observer_k=observer_k,
        observer_gamma=observer_gamma,
    ) if use_adapt else None

    # LQR
    P_lqr, lqr_K = spec["solve_lqr"](
        p, spec["default_lqr_Q"], spec["default_lqr_R"])

    # Cost config (smooth violation replaces binary infeasibility)
    cost_w = dict(spec["default_cost_weights"])
    cost_cfg = CostConfig(
        state_weights=tuple(cost_w["state_weights"]),
        u_weight=cost_w["u_weight"],
        terminal_scale=cost_w["terminal_scale"],
        proj_weight=0.1 if (use_shield or shield_diff) else 0.0,
        infeasible_weight=0.0,
    )

    # Save directory
    if save_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        save_dir = f"runs/{system}_lyap_{ts}"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize policy and Lyapunov (or warm-start from saved run)
    key = jax.random.PRNGKey(seed)

    if warmstart_from is not None:
        ws_path = os.path.join(warmstart_from, "policy_params.pkl")
        with open(ws_path, "rb") as f:
            ws_data = pickle.load(f)
        lyap_cfg = ws_data["lyap_cfg"]
        lyap_params = ws_data["lyap_params"]
        if warmstart_lyap_only:
            # Fresh policy, loaded Lyapunov only
            key, init_key = jax.random.split(key)
            policy_params = init_policy_params(
                init_key, obs_dim, hidden_sizes, out_dim=ctrl_dim)
            print(f"  Warm-started Lyapunov only from: {ws_path}")
            print(f"  Policy initialized from scratch (obs_dim={obs_dim})")
        else:
            policy_params = ws_data["nn"]
            # Expand first layer if obs_dim changed (e.g., adding adaptive inputs)
            ws_obs_dim = ws_data.get("obs_dim", obs_dim)
            if ws_obs_dim != obs_dim:
                W0 = policy_params["layers"][0]["W"]  # (ws_obs_dim, hidden)
                extra = obs_dim - ws_obs_dim
                W0_new = jnp.concatenate(
                    [W0, jnp.zeros((extra, W0.shape[1]))], axis=0)
                layers = list(policy_params["layers"])
                layers[0] = {**layers[0], "W": W0_new}
                policy_params = {**policy_params, "layers": tuple(layers)}
                print(f"  Expanded policy input: {ws_obs_dim} -> {obs_dim} "
                      f"(+{extra} adaptive inputs)")
            print(f"  Warm-started from: {ws_path}")
    else:
        key, init_key = jax.random.split(key)
        policy_params = init_policy_params(
            init_key, obs_dim, hidden_sizes, out_dim=ctrl_dim)
        key, lyap_key = jax.random.split(key)
        lyap_cfg = LyapunovConfig(
            mode="mlp_psd",
            state_dim=state_dim,
            x_eq=spec["x_eq"],
            hidden_sizes=lyap_hidden,
            eps_pd=0.1,
            angle_indices=spec.get("angle_indices", ()),
        )
        lyap_params = init_lyapunov_params(lyap_key, lyap_cfg)

    # Split lyap_params into static and trainable
    lyap_static = {k: v for k, v in lyap_params.items()
                   if k in ("mode", "x_eq")}
    lyap_phi = lyap_params["phi"]

    # CLF config
    clf_cfg = CLFConfig(
        enabled=True, lambda_clf=lambda_clf,
        eps_proj=0.1 if shield_diff else 1e-8,
    )

    # Optimizer
    steps_per_epoch = pool_size // batch_size
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )
    all_params = {"policy": policy_params, "lyap_phi": lyap_phi}
    opt_state = optimizer.init(all_params)

    def _reconstruct_lyap(phi):
        return {**lyap_static, "phi": phi}

    @jax.jit
    def step_fn(all_params, opt_state, batch_x0, batch_a, w_violation):
        def loss_fn(params):
            phi = params["lyap_phi"]
            if freeze_lyap:
                phi = jax.lax.stop_gradient(phi)
            lp = _reconstruct_lyap(phi)
            return _batched_loss_lyap(
                params["policy"], lp, batch_x0, batch_a,
                spec, hidden_sizes, lqr_K, horizon, dt,
                cost_cfg, lyap_cfg, clf_cfg,
                use_shield, lambda_clf, w_violation, policy_mode,
                adapt_cfg=adapt_cfg, shield_diff=shield_diff,
                alpha_max=alpha_max, a_range=a_range,
            )
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(all_params)
        grad_norm = optax.global_norm(grads)
        grads = _sanitize_grads(grads)
        updates, new_opt = optimizer.update(grads, opt_state, all_params)
        new_params = optax.apply_updates(all_params, updates)
        metrics["grad_norm"] = grad_norm
        return new_params, new_opt, loss, metrics

    # Print config
    print(f"\n{'='*60}")
    print(f"  {system} | Neural Lyapunov Training")
    print(f"  Policy mode: {policy_mode}")
    print(f"  {epochs} epochs, horizon={horizon}, dt={dt}")
    print(f"  pool={pool_size}, batch={batch_size}, "
          f"{steps_per_epoch} steps/epoch")
    print(f"  lr={lr}, grad_clip={max_grad_norm}")
    print(f"  lambda_clf={lambda_clf}, w_violation={w_violation_start} -> {w_violation_end}")
    lyap_str = f"MLP-PSD {lyap_hidden}" + (" (FROZEN)" if freeze_lyap else "")
    print(f"  V network: {lyap_str}")
    shield_str = "OFF"
    if use_shield:
        shield_str = "ON (differentiable)" if shield_diff else "ON (stop_gradient)"
    if alpha_max > 0:
        shield_str += f" (alpha_max={alpha_max})"
    print(f"  Shield: {shield_str}")
    print(f"  Region: {region_start} -> {region_end} "
          f"(curriculum over first {curriculum_frac*100:.0f}% of training)")
    a_str = f"a_true={a_true}" if a_range == 0 else f"a ~ U[-{a_range}, {a_range}]"
    print(f"  Uncertainty: {a_str}")
    if use_adapt:
        if use_observer:
            print(f"  Adaptive: OBSERVER (k={observer_k}, gamma={observer_gamma}, obs_dim={obs_dim})")
        else:
            print(f"  Adaptive: HEURISTIC (eta={adapt_eta}, obs_dim={obs_dim})")
    print(f"  Saving to: {save_dir}")
    print(f"{'='*60}\n")

    # Training loop
    history = []
    best_loss = float("inf")
    best_params = all_params
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Region curriculum
        curriculum_end_epoch = max(1, int(epochs * curriculum_frac))
        if epoch <= curriculum_end_epoch:
            progress = epoch / curriculum_end_epoch
            region_scale = region_start + (region_end - region_start) * progress
        else:
            region_scale = region_end

        # Violation weight ramp (high -> low over full training)
        t_frac = (epoch - 1) / max(1, epochs - 1)
        w_violation = w_violation_start + (w_violation_end - w_violation_start) * t_frac
        w_violation_jax = jnp.float32(w_violation)

        # Sample ICs
        key, pool_key, a_key = jax.random.split(key, 3)
        pool_x0, pool_a = spec["sample_ics"](
            pool_key, pool_size, region_scale, a_true)
        if a_range > 0:
            pool_a = jax.random.uniform(
                a_key, (pool_size,), minval=-a_range, maxval=a_range)

        epoch_loss = 0.0
        epoch_grad = 0.0
        epoch_term = 0.0
        epoch_viol = 0.0
        epoch_feas = 0.0
        epoch_V = 0.0

        for step_i in range(steps_per_epoch):
            idx = step_i * batch_size
            batch_x0 = pool_x0[idx:idx + batch_size]
            batch_a = pool_a[idx:idx + batch_size]

            all_params, opt_state, loss, metrics = step_fn(
                all_params, opt_state, batch_x0, batch_a, w_violation_jax)

            epoch_loss += float(loss)
            epoch_grad += float(metrics.get("grad_norm", 0.0))
            epoch_term += float(metrics.get("terminal_norm", 0.0))
            epoch_viol += float(metrics.get("mean_violation", 0.0))
            epoch_feas += float(metrics.get("mean_feasible", 1.0))
            epoch_V += float(metrics.get("mean_V", 0.0))

            global_step = (epoch - 1) * steps_per_epoch + step_i + 1
            record = {"step": global_step, "epoch": epoch,
                      "region_scale": region_scale,
                      "w_violation": w_violation}
            for k, v in metrics.items():
                record[k] = float(v)
            history.append(record)

        n = steps_per_epoch
        avg_loss = epoch_loss / n
        avg_grad = epoch_grad / n
        avg_term = epoch_term / n
        avg_viol = epoch_viol / n
        avg_feas = epoch_feas / n
        avg_V = epoch_V / n

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = jax.tree.map(lambda x: x.copy(), all_params)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            elapsed = time.time() - t0
            feas_str = f"feas {avg_feas:.3f} | " if use_shield else ""
            print(
                f"epoch {epoch:4d}/{epochs} | "
                f"loss {avg_loss:9.3f} | "
                f"term {avg_term:6.2f} | "
                f"viol {avg_viol:8.5f} | "
                f"{feas_str}"
                f"V {avg_V:7.4f} | "
                f"r={region_scale:.2f} | "
                f"wv={w_violation:.1f} | "
                f"grad {avg_grad:8.2f} | "
                f"{elapsed:.1f}s"
            )

    # Extract best params
    save_all = best_params if best_loss < float("inf") else all_params
    policy_final = save_all["policy"]
    lyap_final = _reconstruct_lyap(save_all["lyap_phi"])

    # Save
    save_data = {
        "system": system,
        "nn": jax.device_get(policy_final),
        "lqr_K": jax.device_get(lqr_K),
        "policy_mode": policy_mode,
        "clf_enabled": True,
        "ctrl_dim": ctrl_dim,
        "obs_dim": obs_dim,
        "hidden_sizes": hidden_sizes,
        "lyap_params": jax.device_get(lyap_final),
        "lyap_cfg": lyap_cfg,
        "lambda_clf": lambda_clf,
        "P_lqr": jax.device_get(P_lqr),
        "use_adapt": use_adapt,
        "adapt_cfg": adapt_cfg,
        "shield_diff": shield_diff,
    }

    params_path = os.path.join(save_dir, "policy_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nSaved policy + Lyapunov -> {params_path}")

    hist_path = os.path.join(save_dir, "history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    run_config = {
        "system": system, "policy_mode": policy_mode,
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "pool_size": pool_size, "horizon": horizon, "dt": dt,
        "max_grad_norm": max_grad_norm,
        "lambda_clf": lambda_clf,
        "w_violation_start": w_violation_start,
        "w_violation_end": w_violation_end,
        "lyap_hidden": lyap_hidden, "use_shield": use_shield,
        "region_start": region_start, "region_end": region_end,
        "curriculum_frac": curriculum_frac,
        "a_true": a_true, "a_range": a_range,
        "use_adapt": use_adapt, "adapt_eta": adapt_eta,
        "warmstart_from": warmstart_from,
        "seed": seed,
    }
    config_path = os.path.join(save_dir, "run_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(run_config, f)

    _plot_training_lyap(history, save_dir, use_shield)

    return save_data, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training_lyap(history: list, save_dir: str,
                        use_shield: bool) -> None:
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    viols = [h.get("mean_violation", 0) for h in history]
    norms = [h.get("terminal_norm", 0) for h in history]
    Vs = [h.get("mean_V", 0) for h in history]
    regions = [h.get("region_scale", 0.3) for h in history]

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))

    # Loss
    axes[0, 0].plot(steps, losses)
    axes[0, 0].set(xlabel="step", ylabel="loss", title="Total loss")
    axes[0, 0].set_yscale("symlog", linthresh=1.0)
    axes[0, 0].grid(True, alpha=0.3)

    # Terminal norm
    axes[0, 1].plot(steps, norms, color="tab:orange")
    axes[0, 1].set(xlabel="step", ylabel="|x_T|", title="Terminal norm")
    axes[0, 1].grid(True, alpha=0.3)

    # Smooth violation
    axes[0, 2].plot(steps, viols, color="tab:red")
    axes[0, 2].set(xlabel="step", ylabel="violation",
                    title="CLF violation (smooth)")
    axes[0, 2].set_yscale("symlog", linthresh=1e-6)
    axes[0, 2].grid(True, alpha=0.3)

    # Mean V
    axes[1, 0].plot(steps, Vs, color="tab:purple")
    axes[1, 0].set(xlabel="step", ylabel="V",
                    title="Mean V along trajectories")
    axes[1, 0].grid(True, alpha=0.3)

    # Feasibility (if shield on)
    if use_shield:
        feass = [h.get("mean_feasible", 1.0) for h in history]
        axes[1, 1].plot(steps, feass, color="tab:green")
        axes[1, 1].set(xlabel="step", ylabel="feasibility",
                        title="CLF feasibility")
        axes[1, 1].set_ylim(-0.05, 1.05)
    else:
        axes[1, 1].text(0.5, 0.5, "Shield OFF", transform=axes[1, 1].transAxes,
                         ha="center", va="center", fontsize=14, color="gray")
        axes[1, 1].set(title="CLF feasibility")
    axes[1, 1].grid(True, alpha=0.3)

    # Region scale
    axes[1, 2].plot(steps, regions, color="tab:brown")
    axes[1, 2].set(xlabel="step", ylabel="region_scale",
                    title="Region curriculum")
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(save_dir, "training_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curve -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Joint policy + neural Lyapunov training")
    parser.add_argument("--system", type=str, default="cartpole",
                        choices=list_systems())
    parser.add_argument("--policy-mode", type=str, default="nn_only",
                        choices=["nn_only", "hybrid"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--lambda-clf", type=float, default=0.1)
    parser.add_argument("--w-violation-start", type=float, default=50.0,
                        help="Violation weight at start (high to shape V)")
    parser.add_argument("--w-violation-end", type=float, default=5.0,
                        help="Violation weight at end (low to focus on control)")
    parser.add_argument("--lyap-hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--shield", action="store_true",
                        help="Enable CLF shield projection during training")
    parser.add_argument("--shield-diff", action="store_true",
                        help="Differentiable shield (gradients flow through projection)")
    parser.add_argument("--alpha-max", type=float, default=0.0,
                        help="Cap projection gain in shield (0=no cap)")
    parser.add_argument("--region-start", type=float, default=0.3)
    parser.add_argument("--region-end", type=float, default=0.3)
    parser.add_argument("--curriculum-frac", type=float, default=0.5,
                        help="Fraction of training over which to ramp region")
    parser.add_argument("--a-true", type=float, default=0.0)
    parser.add_argument("--a-range", type=float, default=0.0,
                        help="If >0, randomize a_true ~ U[-a_range, a_range]")
    parser.add_argument("--adapt", action="store_true",
                        help="Enable online adaptive estimator (augments obs)")
    parser.add_argument("--adapt-eta", type=float, default=2e-2,
                        help="Adaptive estimator learning rate")
    parser.add_argument("--observer", action="store_true",
                        help="Use observer-based adaptation (Section 2.1)")
    parser.add_argument("--observer-k", type=float, default=5.0,
                        help="Observer gain (eigenvalue of eta decay)")
    parser.add_argument("--observer-gamma", type=float, default=5.0,
                        help="Observer adaptation gain for a_hat update")
    parser.add_argument("--freeze-lyap", action="store_true",
                        help="Freeze Lyapunov params (train policy only)")
    parser.add_argument("--warmstart-from", type=str, default=None,
                        help="Path to a previous run dir to warm-start from")
    parser.add_argument("--warmstart-lyap-only", action="store_true",
                        help="Load only Lyapunov params from warmstart, fresh policy")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    train_lyapunov(
        system=args.system,
        policy_mode=args.policy_mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        horizon=args.horizon,
        dt=args.dt,
        max_grad_norm=args.max_grad_norm,
        lambda_clf=args.lambda_clf,
        w_violation_start=args.w_violation_start,
        w_violation_end=args.w_violation_end,
        lyap_hidden=tuple(args.lyap_hidden),
        use_shield=args.shield or args.shield_diff,
        shield_diff=args.shield_diff,
        alpha_max=args.alpha_max,
        region_start=args.region_start,
        region_end=args.region_end,
        curriculum_frac=args.curriculum_frac,
        a_true=args.a_true,
        a_range=args.a_range,
        use_adapt=args.adapt or args.observer,
        adapt_eta=args.adapt_eta,
        use_observer=args.observer,
        observer_k=args.observer_k,
        observer_gamma=args.observer_gamma,
        freeze_lyap=args.freeze_lyap,
        warmstart_from=args.warmstart_from,
        warmstart_lyap_only=args.warmstart_lyap_only,
        save_dir=args.save_dir,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
