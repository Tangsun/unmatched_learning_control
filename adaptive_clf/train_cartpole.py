"""Cart-pole training: balance or swing-up via differentiable simulation.

Supports multiple policy modes:
  - hybrid: LQR+NN blend (original, for swing-up)
  - nn_only: pure NN policy (baseline for CLF comparison)
  - clf: NN + CLF shield projection (with fixed or learned Lyapunov)

Usage:
    python -m adaptive_clf.train_cartpole --task swingup --epochs 200
    python -m adaptive_clf.train_cartpole --task balance --epochs 50
    python -m adaptive_clf.train_cartpole --clf --region-scale 0.3 --epochs 100
    python -m adaptive_clf.train_cartpole --learn-lyap --region-scale 0.3 --epochs 100
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
from .nn import init_policy_params, policy_apply
from .cartpole import (
    CartPoleParams, cartpole_affine_terms, cartpole_dynamics,
    rk4_step_cartpole, solve_cartpole_lqr,
)
from .lyapunov import init_lyapunov_params, lyapunov_matrix, lyapunov_value_and_grad
from .shield import clf_shield
from .adaptive import init_adaptive_state, adaptive_update_simple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(theta: Array) -> Array:
    """Wrap angle to [-pi, pi]; gradient is 1 everywhere."""
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def _sanitize_grads(grads):
    """Replace NaN/inf gradients with zero."""
    return jax.tree.map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
        grads,
    )


# ---------------------------------------------------------------------------
# Cost configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CartPoleCost:
    w_theta: float = 5.0
    w_x: float = 1.0
    w_xdot: float = 0.1
    w_thetadot: float = 0.1
    w_u: float = 0.01
    w_terminal: float = 10.0
    w_energy: float = 0.0
    w_proj: float = 0.1        # penalty for CLF projection magnitude
    w_infeasible: float = 50.0  # penalty for CLF infeasibility


def _pole_energy(x: Array, p: CartPoleParams) -> Array:
    """Pole energy relative to upright equilibrium (E=0 at upright, E<0 hanging)."""
    theta, thetadot = x[1], x[3]
    return 0.5 * p.mp * p.l ** 2 * thetadot ** 2 + p.mp * p.g * p.l * (jnp.cos(theta) - 1.0)


def _stage_cost(x: Array, u: Array, u_nom: Array, feasible: Array,
                cfg: CartPoleCost, p: CartPoleParams) -> Array:
    """Per-step cost using wrapped theta^2 (nonzero gradient everywhere)."""
    theta_w = _wrap_angle(x[1])
    cost = (
        cfg.w_theta * theta_w ** 2
        + cfg.w_x * x[0] ** 2
        + cfg.w_xdot * x[2] ** 2
        + cfg.w_thetadot * x[3] ** 2
        + cfg.w_u * u ** 2
        + cfg.w_proj * (u - u_nom) ** 2
        + cfg.w_infeasible * (1.0 - feasible)
    )
    if cfg.w_energy > 0:
        E = _pole_energy(x, p)
        cost = cost + cfg.w_energy * E ** 2
    return cost


def _terminal_cost(xT: Array, cfg: CartPoleCost, p: CartPoleParams) -> Array:
    theta_w = _wrap_angle(xT[1])
    cost = cfg.w_terminal * (
        theta_w ** 2
        + 0.1 * xT[0] ** 2
        + 0.1 * xT[2] ** 2
        + 0.1 * xT[3] ** 2
    )
    if cfg.w_energy > 0:
        E = _pole_energy(xT, p)
        cost = cost + cfg.w_terminal * cfg.w_energy * E ** 2
    return cost


# ---------------------------------------------------------------------------
# Observation & policy modes
# ---------------------------------------------------------------------------

def _make_obs(x: Array) -> Array:
    """[x_cart, sin(theta), cos(theta)-1, xdot, thetadot] -- 5D.

    Centered so observation is all-zeros at the upright equilibrium.
    """
    return jnp.array([x[0], jnp.sin(x[1]), jnp.cos(x[1]) - 1.0, x[2], x[3]])


OBS_DIM = 5


def _nn_policy(
    policy_params: Dict[str, Any],
    x: Array,
    p: CartPoleParams,
    hidden_sizes: Tuple[int, ...],
) -> Array:
    """Pure NN policy (no LQR blend)."""
    obs = _make_obs(x)
    return policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes)


def _hybrid_policy(
    policy_params: Dict[str, Any],
    x: Array,
    lqr_K: Array,
    p: CartPoleParams,
    hidden_sizes: Tuple[int, ...],
) -> Array:
    """Blend LQR (near upright) and NN (far from upright) with smooth sigmoid."""
    obs = _make_obs(x)
    u_nn = policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes)
    u_lqr = jnp.clip((-lqr_K @ x).squeeze(), p.u_min, p.u_max)
    alpha = jax.nn.sigmoid(8.0 * (jnp.cos(x[1]) - 0.5))
    return alpha * u_lqr + (1.0 - alpha) * u_nn


# ---------------------------------------------------------------------------
# Rollout (unified: supports hybrid, nn_only, clf modes)
# ---------------------------------------------------------------------------

def _episode_rollout(
    policy_params: Dict[str, Any],
    lyap_params: Dict[str, Any] | None,
    x0: Array,
    a_true: Array,
    p: CartPoleParams,
    hidden_sizes: Tuple[int, ...],
    lqr_K: Array,
    horizon: int,
    dt: float,
    cost_cfg: CartPoleCost,
    clf_cfg: CLFConfig | None,
    lyap_cfg: LyapunovConfig | None,
    adapt_cfg: AdaptiveConfig | None,
    policy_mode: str,
) -> Tuple[Array, Dict[str, Array]]:
    x_max = 20.0
    use_clf = (policy_mode == "clf" and clf_cfg is not None
               and clf_cfg.enabled and lyap_params is not None)
    use_adapt = (adapt_cfg is not None and adapt_cfg.adapt_enabled)

    if use_adapt:
        adaptive_state0 = init_adaptive_state(p, adapt_cfg)
    else:
        adaptive_state0 = AdaptiveState(
            a_hat=jnp.array(0.0), info=jnp.array(1e-6), radius=jnp.array(0.0))

    def body(carry, _):
        x_raw, adaptive_state = carry
        x = jnp.clip(x_raw, -x_max, x_max)
        x = x.at[1].set(_wrap_angle(x[1]))

        # Compute nominal control
        if policy_mode == "hybrid":
            u_nom = _hybrid_policy(policy_params, x, lqr_K, p, hidden_sizes)
        else:
            u_nom = _nn_policy(policy_params, x, p, hidden_sizes)

        # Optionally apply CLF shield
        if use_clf:
            u_shield, shield_aux = clf_shield(
                u_nom=u_nom,
                x=x,
                adaptive_state=adaptive_state,
                lyap_params=lyap_params,
                lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg,
                p=p,
                affine_terms_fn=cartpole_affine_terms,
                input_bounds=(p.u_min, p.u_max),
            )
            u = jnp.clip(u_shield, p.u_min, p.u_max)
            feasible = shield_aux.get("feasible", jnp.array(1.0))
            V_val = shield_aux["V"]
        else:
            u = u_nom
            feasible = jnp.array(1.0)
            V_val = jnp.array(0.0)

        # Adaptive update
        if use_adapt:
            adaptive_next = adaptive_update_simple(
                adaptive_state, x, u, a_true, dt, p, adapt_cfg,
                affine_terms_fn=cartpole_affine_terms,
                dynamics_fn=lambda x_, u_, a_, p_: cartpole_dynamics(x_, u_, a_, p_),
            )
        else:
            adaptive_next = adaptive_state

        x_next = rk4_step_cartpole(x, u, a_true, dt, p)
        cost = _stage_cost(x, u, u_nom, feasible, cost_cfg, p)
        history = {
            "x": x, "u": u, "u_nom": u_nom, "cost": cost,
            "V": V_val, "feasible": feasible,
            "a_hat": adaptive_state.a_hat, "radius": adaptive_state.radius,
        }
        return (x_next, adaptive_next), history

    (xT, adaptive_T), hist = jax.lax.scan(body, (x0, adaptive_state0), jnp.arange(horizon))
    xT = xT.at[1].set(_wrap_angle(xT[1]))

    mean_stage = jnp.mean(hist["cost"])
    terminal = _terminal_cost(xT, cost_cfg, p)
    total_loss = jnp.nan_to_num(mean_stage + terminal, nan=1e4)

    metrics = {
        "loss": total_loss,
        "terminal_norm": jnp.linalg.norm(xT),
        "mean_stage_cost": mean_stage,
        "terminal_cost": terminal,
        "mean_feasible": jnp.mean(hist["feasible"]),
        "mean_V": jnp.mean(hist["V"]),
        "mean_proj": jnp.mean(jnp.abs(hist["u"] - hist["u_nom"])),
    }
    return total_loss, metrics


def _batched_loss(
    policy_params, lyap_params, batch_x0, batch_a, p, hidden_sizes, lqr_K,
    horizon, dt, cost_cfg, clf_cfg, lyap_cfg, adapt_cfg, policy_mode,
):
    fn = lambda x0, a: _episode_rollout(
        policy_params, lyap_params, x0, a, p, hidden_sizes, lqr_K,
        horizon, dt, cost_cfg, clf_cfg, lyap_cfg, adapt_cfg, policy_mode,
    )
    losses, metrics = jax.vmap(fn)(batch_x0, batch_a)
    safe = jnp.nan_to_num(losses, nan=1e4)
    return jnp.mean(safe), {k: jnp.nan_to_num(jnp.mean(v), nan=0.0) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_cartpole_batch(
    key: Array,
    batch_size: int,
    task: str = "swingup",
    a_true: float = 0.0,
    region_scale: float = 1.0,
) -> Tuple[Array, Array]:
    if task == "balance":
        low = jnp.array([-0.5, -0.3, -0.5, -0.5])
        high = jnp.array([0.5, 0.3, 0.5, 0.5])
    else:
        low = jnp.array([-1.0, -jnp.pi, -1.0, -1.0])
        high = jnp.array([1.0, jnp.pi, 1.0, 1.0])
    low = low * region_scale
    high = high * region_scale
    batch_x0 = jax.random.uniform(key, (batch_size, 4), minval=low, maxval=high)
    batch_a = jnp.full((batch_size,), a_true)
    return batch_x0, batch_a


def sample_cartpole_curriculum_batch(
    key: Array,
    batch_size: int,
    progress: float,
    a_true: float = 0.0,
    region_scale: float = 1.0,
) -> Tuple[Array, Array]:
    """Curriculum ICs with 50% balance ICs to maintain stabilization quality."""
    k_bal, k_cur = jax.random.split(key)
    n_balance = max(1, batch_size // 2)
    n_curriculum = batch_size - n_balance

    bal_low = jnp.array([-0.5, -0.3, -0.5, -0.5]) * region_scale
    bal_high = jnp.array([0.5, 0.3, 0.5, 0.5]) * region_scale
    batch_balance = jax.random.uniform(k_bal, (n_balance, 4), minval=bal_low, maxval=bal_high)

    theta_range = region_scale * (0.5 + progress * (jnp.pi - 0.5))
    x_range = region_scale * (0.5 + progress * 0.5)
    v_range = region_scale * (0.5 + progress * 1.0)
    cur_low = jnp.array([-x_range, -theta_range, -v_range, -v_range])
    cur_high = jnp.array([x_range, theta_range, v_range, v_range])
    batch_curriculum = jax.random.uniform(k_cur, (n_curriculum, 4), minval=cur_low, maxval=cur_high)

    batch_x0 = jnp.concatenate([batch_balance, batch_curriculum], axis=0)
    batch_a = jnp.full((batch_size,), a_true)
    return batch_x0, batch_a


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _default_run_dir(task: str, epochs: int, lr: float,
                     batch_size: int, pool_size: int, horizon: int,
                     policy_mode: str) -> str:
    ts = datetime.now().strftime("%m%d_%H%M")
    return (f"runs/cartpole_{task}_{policy_mode}_{ts}_ep{epochs}_h{horizon}"
            f"_pool{pool_size}_bs{batch_size}_lr{lr}")


def train_cartpole(
    task: str = "swingup",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    pool_size: int = 2048,
    horizon: int = 200,
    dt: float = 0.02,
    max_grad_norm: float = 10.0,
    curriculum_frac: float = 0.5,
    log_every: int = 5,
    save_dir: str | None = None,
    seed: int = 0,
    # CLF options
    clf_enabled: bool = False,
    learn_lyap: bool = False,
    lambda_clf: float = 0.5,
    region_scale: float = 1.0,
    # Policy mode
    policy_mode: str | None = None,
) -> Tuple[Dict[str, Any], list]:
    """Train a cart-pole controller and return (policy_params, history).

    policy_mode:
      - "hybrid": LQR+NN blend (default for swingup without CLF)
      - "nn_only": pure NN (default for balance or CLF experiments)
      - "clf": NN + CLF shield projection
    """
    if learn_lyap:
        clf_enabled = True
    if policy_mode is None:
        if clf_enabled:
            policy_mode = "clf"
        elif task == "balance":
            policy_mode = "nn_only"
        else:
            policy_mode = "hybrid"

    p = CartPoleParams()

    Q_lqr = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R_lqr = jnp.array([[0.01]])
    P_lqr, lqr_K = solve_cartpole_lqr(p, Q_lqr, R_lqr)
    print(f"LQR gain K = {lqr_K}")

    # Cost config
    if task == "balance":
        cost_cfg = CartPoleCost(
            w_theta=10.0, w_x=1.0, w_xdot=1.0, w_thetadot=1.0,
            w_u=0.01, w_terminal=5.0,
        )
    elif clf_enabled:
        cost_cfg = CartPoleCost(
            w_theta=10.0, w_x=1.0, w_xdot=0.5, w_thetadot=0.5,
            w_u=0.01, w_terminal=10.0, w_energy=0.0,
            w_proj=0.1, w_infeasible=50.0,
        )
    else:
        cost_cfg = CartPoleCost(
            w_theta=2.0, w_x=0.5, w_xdot=0.1, w_thetadot=0.1,
            w_u=0.05, w_terminal=10.0, w_energy=0.5,
        )

    if save_dir is None:
        save_dir = _default_run_dir(task, epochs, lr, batch_size, pool_size,
                                     horizon, policy_mode)
    os.makedirs(save_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    hidden_sizes = (64, 64)
    key, init_key = jax.random.split(key)
    policy_params = init_policy_params(init_key, obs_dim=OBS_DIM, hidden_sizes=hidden_sizes)

    # CLF / Lyapunov setup
    clf_cfg = CLFConfig(enabled=clf_enabled, lambda_clf=lambda_clf)
    adapt_cfg = AdaptiveConfig(adapt_enabled=False)

    if clf_enabled:
        key, lyap_key = jax.random.split(key)
        if learn_lyap:
            lyap_cfg = LyapunovConfig(
                mode="quadratic_learned",
                state_dim=4,
                x_eq=(0.0, 0.0, 0.0, 0.0),
                P_init=P_lqr,
            )
        else:
            lyap_cfg = LyapunovConfig(
                mode="quadratic_fixed",
                state_dim=4,
                x_eq=(0.0, 0.0, 0.0, 0.0),
                P_init=P_lqr,
            )
        lyap_params = init_lyapunov_params(lyap_key, lyap_cfg)
    else:
        lyap_cfg = None
        lyap_params = None

    steps_per_epoch = pool_size // batch_size
    total_steps = epochs * steps_per_epoch

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )

    # Build optimizable params
    if learn_lyap:
        lyap_static = {k: v for k, v in lyap_params.items() if not hasattr(v, 'shape')}
        lyap_trainable = {k: v for k, v in lyap_params.items() if hasattr(v, 'shape')}
        all_params = {"policy": policy_params, "lyap": lyap_trainable}
        opt_state = optimizer.init(all_params)
    else:
        all_params = policy_params
        opt_state = optimizer.init(all_params)

    use_curriculum = (task == "swingup" and region_scale >= 1.0)
    curriculum_end = max(1, int(curriculum_frac * epochs)) if use_curriculum else 0

    # Build step function
    if learn_lyap:
        @jax.jit
        def step_fn(all_params, opt_state, batch_x0, batch_a):
            def loss_fn(params):
                full_lyap = {**lyap_static, **params["lyap"]}
                return _batched_loss(
                    params["policy"], full_lyap, batch_x0, batch_a, p, hidden_sizes,
                    lqr_K, horizon, dt, cost_cfg, clf_cfg, lyap_cfg, adapt_cfg, policy_mode,
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
                    pp, lyap_params, batch_x0, batch_a, p, hidden_sizes,
                    lqr_K, horizon, dt, cost_cfg, clf_cfg, lyap_cfg, adapt_cfg, policy_mode,
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
    print(f"  Cart-pole {task} | mode={policy_mode}")
    print(f"  {epochs} epochs, horizon={horizon}, dt={dt}")
    print(f"  pool={pool_size}, batch={batch_size}, "
          f"{steps_per_epoch} steps/epoch, {total_steps} total steps")
    print(f"  lr={lr}, grad_clip={max_grad_norm}")
    if clf_enabled:
        mode_str = "learned P" if learn_lyap else "fixed P_lqr"
        print(f"  CLF: ON (lambda={lambda_clf}, {mode_str})")
    else:
        print(f"  CLF: OFF")
    print(f"  Region scale: {region_scale}")
    if use_curriculum:
        print(f"  Curriculum: expand to full range over {curriculum_end} epochs")
    print(f"  Saving to: {save_dir}")
    print(f"{'='*60}\n")

    history = []
    t0 = time.time()
    global_step = 0
    best_loss = float("inf")
    best_params = all_params

    for epoch in range(1, epochs + 1):
        key, pool_key = jax.random.split(key)

        if use_curriculum:
            cur_progress = min(1.0, (epoch - 1) / max(1, curriculum_end - 1))
            pool_x0, pool_a = sample_cartpole_curriculum_batch(
                pool_key, pool_size, cur_progress, region_scale=region_scale,
            )
            phase = f"cur {cur_progress:.2f}" if cur_progress < 1.0 else "full"
        else:
            pool_x0, pool_a = sample_cartpole_batch(
                pool_key, pool_size, task, region_scale=region_scale,
            )
            phase = f"r={region_scale:.1f}"

        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, pool_size)
        pool_x0 = pool_x0[perm]
        pool_a = pool_a[perm]

        epoch_loss = 0.0
        epoch_grad = 0.0
        epoch_term = 0.0
        epoch_feas = 0.0

        for i in range(steps_per_epoch):
            global_step += 1
            start = i * batch_size
            batch_x0 = pool_x0[start : start + batch_size]
            batch_a = pool_a[start : start + batch_size]

            all_params, opt_state, loss, metrics = step_fn(
                all_params, opt_state, batch_x0, batch_a,
            )

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
            best_params = jax.tree.map(lambda x: x.copy(), all_params)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            elapsed = time.time() - t0
            best_str = f"best {best_loss:8.4f}" if best_loss < float("inf") else "best      ---"
            feas_str = f"feas {avg_feas:.3f} | " if clf_enabled else ""
            lyap_str = ""
            if learn_lyap:
                cur_lyap = best_params if isinstance(best_params, dict) and "lyap" in best_params else all_params
                if isinstance(cur_lyap, dict) and "lyap" in cur_lyap:
                    P_cur = lyapunov_matrix({**lyap_static, **cur_lyap["lyap"]}, lyap_cfg)
                    eigs = jnp.linalg.eigvalsh(P_cur)
                    lyap_str = f" | P_eig [{float(eigs[0]):.2f},{float(eigs[-1]):.2f}]"
            print(
                f"epoch {epoch:4d}/{epochs} "
                f"[{phase:8s}] | "
                f"loss {avg_loss:9.4f} | "
                f"term_norm {avg_term:7.3f} | "
                f"{feas_str}"
                f"grad {avg_grad:8.4f} | "
                f"{best_str}{lyap_str} | "
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
        "nn": jax.device_get(policy_final),
        "lqr_K": jax.device_get(lqr_K),
        "policy_mode": policy_mode,
        "clf_enabled": clf_enabled,
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
        "task": task, "policy_mode": policy_mode,
        "clf_enabled": clf_enabled, "learn_lyap": learn_lyap,
        "lambda_clf": lambda_clf, "region_scale": region_scale,
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "pool_size": pool_size, "horizon": horizon, "dt": dt,
        "max_grad_norm": max_grad_norm, "seed": seed,
    }
    config_path = os.path.join(save_dir, "run_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(run_config, f)

    _plot_training(history, save_dir, clf_enabled)
    _eval_policy(policy_final, p, hidden_sizes, lqr_K, task, horizon, dt,
                 cost_cfg, save_dir, policy_mode, lyap_final, lyap_cfg, clf_cfg,
                 adapt_cfg, region_scale=region_scale)

    if learn_lyap:
        P_final = lyapunov_matrix(lyap_final, lyap_cfg)
        eigs = jnp.linalg.eigvalsh(P_final)
        print(f"Learned P eigenvalues: {eigs}")

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
    ax1 = axes[0]
    ax2 = axes[1]

    ax1.plot(steps, losses)
    ax1.set(xlabel="step", ylabel="loss", title="Training loss")
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, norms, color="tab:orange")
    ax2.set(xlabel="step", ylabel="|x_T|", title="Terminal state norm")
    ax2.grid(True, alpha=0.3)

    if clf_enabled and ncols == 3:
        ax3 = axes[2]
        feas = [h.get("mean_feasible", 1.0) for h in history]
        ax3.plot(steps, feas, color="tab:green")
        ax3.set(xlabel="step", ylabel="feasibility", title="CLF feasibility")
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(save_dir, "training_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curve -> {out}")


def _eval_policy(policy_params, p, hidden_sizes, lqr_K, task, horizon, dt,
                 cost_cfg, save_dir, policy_mode, lyap_params, lyap_cfg,
                 clf_cfg, adapt_cfg, region_scale=1.0) -> None:
    """Simulate from test ICs: 3 inside training region + 2 outside."""
    eval_horizon = max(horizon, 400)
    if task == "balance":
        ics = [
            jnp.array([0.0, 0.2, 0.0, 0.0]),
            jnp.array([0.0, -0.2, 0.0, 0.0]),
            jnp.array([0.5, 0.1, 0.0, 0.0]),
        ]
        labels = ["theta0=0.2", "theta0=-0.2", "x0=0.5, theta0=0.1"]
    else:
        # 3 ICs inside training region, 2 outside
        theta_in = region_scale * jnp.pi  # max theta in training region
        ics = [
            # Inside training region
            jnp.array([0.0, 0.5 * theta_in, 0.0, 0.0]),
            jnp.array([0.0, -theta_in, 0.0, 0.0]),
            jnp.array([0.2, 0.8 * theta_in, 0.3, 0.3]),
            # Outside training region
            jnp.array([0.0, jnp.pi, 0.0, 0.0]),
            jnp.array([0.5, jnp.pi, 0.0, 0.5]),
        ]
        deg_in = int(jnp.degrees(theta_in))
        labels = [
            f"IN: theta={int(jnp.degrees(0.5*theta_in))}deg",
            f"IN: theta=-{deg_in}deg",
            f"IN: x=0.2, theta={int(jnp.degrees(0.8*theta_in))}deg, v=0.3",
            "OUT: theta=180deg (hanging)",
            "OUT: x=0.5, theta=180deg, v=0.5",
        ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_x, ax_theta, ax_vel, ax_u = axes.flat
    ts = jnp.arange(eval_horizon) * dt

    for ic, label in zip(ics, labels):
        xs, us = _simulate(policy_params, ic, p, hidden_sizes, lqr_K,
                           eval_horizon, dt, policy_mode, lyap_params,
                           lyap_cfg, clf_cfg, adapt_cfg)
        ax_x.plot(ts, xs[:, 0], label=label)
        ax_theta.plot(ts, jnp.degrees(xs[:, 1]), label=label)
        ax_vel.plot(ts, xs[:, 2], label=label, linestyle="-")
        ax_vel.plot(ts, xs[:, 3], label=None, linestyle="--", alpha=0.5)
        ax_u.plot(ts, us, label=label)

    ax_x.set(ylabel="x_cart", title="Cart position")
    ax_theta.set(ylabel="theta (deg)", title="Pole angle")
    ax_theta.axhline(0, color="k", ls=":", lw=0.8)
    ax_vel.set(ylabel="velocity", title="Velocities (solid=cart, dashed=pole)")
    ax_u.set(ylabel="u", title="Control force")

    for ax in axes.flat:
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    mode_str = {"hybrid": "hybrid LQR+NN", "nn_only": "NN only", "clf": "NN+CLF"}
    fig.suptitle(f"Cart-pole {task} evaluation ({mode_str.get(policy_mode, policy_mode)})",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(save_dir, "eval_cartpole.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved eval plot -> {out}")


def _simulate(policy_params, x0, p, hidden_sizes, lqr_K, horizon, dt,
              policy_mode="hybrid", lyap_params=None, lyap_cfg=None,
              clf_cfg=None, adapt_cfg=None):
    """Roll out the policy and collect trajectory."""
    a_true = jnp.array(0.0)
    x_max = 20.0
    use_clf = (policy_mode == "clf" and clf_cfg is not None
               and clf_cfg.enabled and lyap_params is not None)
    use_adapt = (adapt_cfg is not None and adapt_cfg.adapt_enabled)

    if use_adapt:
        adaptive_state0 = init_adaptive_state(p, adapt_cfg)
    else:
        adaptive_state0 = AdaptiveState(
            a_hat=jnp.array(0.0), info=jnp.array(1e-6), radius=jnp.array(0.0))

    def body(carry, _):
        x_raw, adaptive_state = carry
        x = jnp.clip(x_raw, -x_max, x_max)
        x = x.at[1].set(_wrap_angle(x[1]))

        if policy_mode == "hybrid":
            u_nom = _hybrid_policy(policy_params, x, lqr_K, p, hidden_sizes)
        else:
            u_nom = _nn_policy(policy_params, x, p, hidden_sizes)

        if use_clf:
            u_shield, _ = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=adaptive_state,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p,
                affine_terms_fn=cartpole_affine_terms,
                input_bounds=(p.u_min, p.u_max),
            )
            u = jnp.clip(u_shield, p.u_min, p.u_max)
        else:
            u = u_nom

        if use_adapt:
            adaptive_next = adaptive_update_simple(
                adaptive_state, x, u, a_true, dt, p, adapt_cfg,
                affine_terms_fn=cartpole_affine_terms,
                dynamics_fn=lambda x_, u_, a_, p_: cartpole_dynamics(x_, u_, a_, p_),
            )
        else:
            adaptive_next = adaptive_state

        x_next = rk4_step_cartpole(x, u, a_true, dt, p)
        return (x_next, adaptive_next), (x, u)

    _, (xs, us) = jax.lax.scan(body, (x0, adaptive_state0), jnp.arange(horizon))
    return xs, us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train cart-pole controller")
    parser.add_argument("--task", choices=["balance", "swingup"], default="swingup")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--curriculum-frac", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=5,
                        help="Log every N epochs")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # CLF options
    parser.add_argument("--clf", action="store_true", default=False,
                        help="Enable CLF shield (fixed P_lqr)")
    parser.add_argument("--learn-lyap", action="store_true", default=False,
                        help="Jointly learn Lyapunov P (implies --clf)")
    parser.add_argument("--lambda-clf", type=float, default=0.5,
                        help="CLF decay rate")
    parser.add_argument("--region-scale", type=float, default=1.0,
                        help="Scale IC region (0.3 = ~55 deg from upright)")
    parser.add_argument("--policy-mode", choices=["hybrid", "nn_only", "clf"],
                        default=None, help="Policy mode (auto-detected if omitted)")
    args = parser.parse_args()

    train_cartpole(
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        horizon=args.horizon,
        dt=args.dt,
        max_grad_norm=args.max_grad_norm,
        curriculum_frac=args.curriculum_frac,
        log_every=args.log_every,
        save_dir=args.save_dir,
        seed=args.seed,
        clf_enabled=args.clf,
        learn_lyap=args.learn_lyap,
        lambda_clf=args.lambda_clf,
        region_scale=args.region_scale,
        policy_mode=args.policy_mode,
    )


if __name__ == "__main__":
    main()
