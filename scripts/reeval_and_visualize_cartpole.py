"""Re-evaluate saved cart-pole policies with updated ICs and generate visualizations.

Usage:
    conda activate dqs_jax
    python scripts/reeval_and_visualize_cartpole.py
"""

import os
import pickle
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_clf.configs import AdaptiveConfig, AdaptiveState, CLFConfig, LyapunovConfig
from adaptive_clf.nn import init_policy_params, policy_apply
from adaptive_clf.cartpole import (
    CartPoleParams, cartpole_affine_terms, cartpole_dynamics,
    rk4_step_cartpole, solve_cartpole_lqr,
)
from adaptive_clf.lyapunov import init_lyapunov_params, lyapunov_value, lyapunov_value_and_grad, lyapunov_matrix
from adaptive_clf.shield import clf_shield
from adaptive_clf.adaptive import init_adaptive_state


# ---------------------------------------------------------------------------
# Policy functions (matching train_cartpole.py)
# ---------------------------------------------------------------------------

OBS_DIM = 5

def _wrap_angle(theta):
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))

def _make_obs(x):
    """[x_cart, sin(theta), cos(theta)-1, xdot, thetadot] -- matches train_cartpole."""
    return jnp.array([x[0], jnp.sin(x[1]), jnp.cos(x[1]) - 1.0, x[2], x[3]])

def _nn_policy(policy_params, x, p, hidden_sizes):
    obs = _make_obs(x)
    return policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes)

def _hybrid_policy(policy_params, x, lqr_K, p, hidden_sizes):
    obs = _make_obs(x)
    u_nn = policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes)
    u_lqr = jnp.clip((-lqr_K @ x).squeeze(), p.u_min, p.u_max)
    alpha = jax.nn.sigmoid(8.0 * (jnp.cos(x[1]) - 0.5))
    return alpha * u_lqr + (1.0 - alpha) * u_nn


def simulate(policy_params, x0, p, hidden_sizes, lqr_K, horizon, dt,
             policy_mode, lyap_params, lyap_cfg, clf_cfg, adapt_cfg):
    """Roll out and return (xs, us, Vs, shield_info)."""
    a_true = jnp.array(0.0)
    x_max = 20.0
    use_clf = (policy_mode == "clf" and clf_cfg is not None
               and clf_cfg.enabled and lyap_params is not None)

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
            u_shield, aux = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=adaptive_state,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p,
                affine_terms_fn=cartpole_affine_terms,
            )
            u = jnp.clip(u_shield, p.u_min, p.u_max)
            V = aux["V"]
            violation = aux["violation"]
        else:
            u = u_nom
            # Compute V even without CLF for visualization
            if lyap_params is not None and lyap_cfg is not None:
                V, _ = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
            else:
                V = jnp.array(0.0)
            violation = jnp.array(0.0)

        x_next = rk4_step_cartpole(x, u, a_true, dt, p)
        return (x_next, adaptive_state), (x, u, u_nom, V, violation)

    _, (xs, us, us_nom, Vs, violations) = jax.lax.scan(
        body, (x0, adaptive_state0), jnp.arange(horizon))
    return xs, us, us_nom, Vs, violations


def load_run(run_dir):
    """Load policy params and config from a run directory."""
    with open(os.path.join(run_dir, "policy_params.pkl"), "rb") as f:
        save_data = pickle.load(f)
    with open(os.path.join(run_dir, "run_config.pkl"), "rb") as f:
        run_config = pickle.load(f)
    return save_data, run_config


def make_eval_ics(region_scale):
    """Generate eval ICs: 3 inside training region, 2 outside."""
    theta_in = region_scale * jnp.pi
    ics = [
        jnp.array([0.0, 0.5 * theta_in, 0.0, 0.0]),
        jnp.array([0.0, -theta_in, 0.0, 0.0]),
        jnp.array([0.2, 0.8 * theta_in, 0.3, 0.3]),
        jnp.array([0.0, jnp.pi, 0.0, 0.0]),
        jnp.array([0.5, jnp.pi, 0.0, 0.5]),
    ]
    labels = [
        f"IN: theta={0.5*theta_in:.2f}",
        f"IN: theta=-{float(theta_in):.2f}",
        f"IN: x=0.2, theta={0.8*theta_in:.2f}, v=0.3",
        "OUT: theta=pi (hanging)",
        "OUT: x=0.5, theta=pi, v=0.5",
    ]
    return ics, labels


def plot_eval(run_dir, run_name, save_data, run_config):
    """Standard 2x2 eval plot with state/control trajectories."""
    p = CartPoleParams()
    hidden_sizes = (64, 64)
    Q_lqr = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R_lqr = jnp.array([[0.01]])
    P_lqr, lqr_K = solve_cartpole_lqr(p, Q_lqr, R_lqr)

    policy_params = save_data["nn"]
    policy_mode = save_data["policy_mode"]
    clf_enabled = save_data.get("clf_enabled", False)
    region_scale = run_config.get("region_scale", 1.0)
    lambda_clf = run_config.get("lambda_clf", 0.5)
    dt = run_config.get("dt", 0.02)
    horizon = max(run_config.get("horizon", 200), 400)

    # Setup CLF/Lyapunov
    if clf_enabled and "lyap_params" in save_data:
        learn_lyap = run_config.get("learn_lyap", False)
        lyap_cfg = LyapunovConfig(
            mode="quadratic_learned" if learn_lyap else "quadratic_fixed",
            state_dim=4,
            x_eq=(0.0, 0.0, 0.0, 0.0),
            P_init=P_lqr,
        )
        lyap_params = save_data["lyap_params"]
        clf_cfg = CLFConfig(enabled=True, lambda_clf=lambda_clf)
    else:
        lyap_cfg = None
        lyap_params = None
        clf_cfg = CLFConfig(enabled=False)

    adapt_cfg = AdaptiveConfig(adapt_enabled=False)
    ics, labels = make_eval_ics(region_scale)

    ts = jnp.arange(horizon) * dt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_x, ax_theta, ax_vel, ax_u = axes.flat

    colors = plt.cm.tab10(range(len(ics)))

    for ic, label, c in zip(ics, labels, colors):
        xs, us, us_nom, Vs, violations = simulate(
            policy_params, ic, p, hidden_sizes, lqr_K, horizon, dt,
            policy_mode, lyap_params, lyap_cfg, clf_cfg, adapt_cfg)

        is_outside = label.startswith("OUT:")
        ls = "--" if is_outside else "-"
        lw = 1.5 if is_outside else 2.0
        alpha = 0.7 if is_outside else 1.0

        ax_x.plot(ts, xs[:, 0], label=label, color=c, ls=ls, lw=lw, alpha=alpha)
        ax_theta.plot(ts, xs[:, 1], label=label, color=c, ls=ls, lw=lw, alpha=alpha)
        ax_vel.plot(ts, xs[:, 2], color=c, ls=ls, lw=lw, alpha=alpha, label=label)
        ax_vel.plot(ts, xs[:, 3], color=c, ls=":", lw=1.0, alpha=0.4)
        ax_u.plot(ts, us, label=label, color=c, ls=ls, lw=lw, alpha=alpha)

    ax_x.set(ylabel="x_cart", title="Cart position")
    ax_theta.set(ylabel="theta (rad)", title="Pole angle (0=upright)")
    ax_theta.axhline(0, color="k", ls=":", lw=0.8)
    theta_bound = region_scale * jnp.pi
    ax_theta.axhspan(-theta_bound, theta_bound,
                     alpha=0.08, color="green", label=f"training region (±{float(theta_bound):.2f} rad)")
    ax_vel.set(ylabel="velocity", title="Velocities (solid=cart, :=pole)")
    ax_u.set(ylabel="u", title="Control force")

    for ax in axes.flat:
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    mode_labels = {"hybrid": "hybrid LQR+NN", "nn_only": "NN only", "clf": "NN+CLF"}
    clf_str = ""
    if clf_enabled:
        clf_str = f", learned P" if run_config.get("learn_lyap") else f", fixed P_lqr"
    fig.suptitle(f"{run_name}: {mode_labels.get(policy_mode, policy_mode)}{clf_str}, "
                 f"region={region_scale}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(run_dir, "eval_cartpole.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved eval plot -> {out}")


def plot_lyapunov_and_shield(run_dir, run_name, save_data, run_config):
    """Phase portrait with V contours and shield activity (CLF runs only)."""
    clf_enabled = save_data.get("clf_enabled", False)
    if not clf_enabled or "lyap_params" not in save_data:
        print(f"  Skipping Lyapunov visualization (no CLF) for {run_name}")
        return

    p = CartPoleParams()
    hidden_sizes = (64, 64)
    Q_lqr = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R_lqr = jnp.array([[0.01]])
    P_lqr, lqr_K = solve_cartpole_lqr(p, Q_lqr, R_lqr)

    policy_params = save_data["nn"]
    policy_mode = save_data["policy_mode"]
    region_scale = run_config.get("region_scale", 1.0)
    lambda_clf = run_config.get("lambda_clf", 0.5)
    learn_lyap = run_config.get("learn_lyap", False)
    dt = run_config.get("dt", 0.02)
    horizon = 400

    lyap_cfg = LyapunovConfig(
        mode="quadratic_learned" if learn_lyap else "quadratic_fixed",
        state_dim=4, x_eq=(0.0, 0.0, 0.0, 0.0), P_init=P_lqr)
    lyap_params = save_data["lyap_params"]
    clf_cfg = CLFConfig(enabled=True, lambda_clf=lambda_clf)
    adapt_cfg = AdaptiveConfig(adapt_enabled=False)

    # --- Plot 1: V(theta, thetadot) contour with trajectories ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # V contour in (theta, thetadot) plane (x=0, xdot=0)
    theta_range = jnp.linspace(-jnp.pi, jnp.pi, 100)
    thdot_range = jnp.linspace(-3.0, 3.0, 100)
    TH, THD = jnp.meshgrid(theta_range, thdot_range)

    # Build grid of states [0, theta, 0, thetadot] and compute V
    TH_flat = TH.ravel()
    THD_flat = THD.ravel()
    X_grid = jnp.stack([jnp.zeros_like(TH_flat), TH_flat,
                        jnp.zeros_like(TH_flat), THD_flat], axis=1)

    @jax.vmap
    def compute_V_batch(x):
        return lyapunov_value(lyap_params, lyap_cfg, x)

    V_grid = compute_V_batch(X_grid).reshape(TH.shape)

    ax = axes[0]
    levels = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    cs = ax.contourf(TH, THD, V_grid, levels=20, cmap="RdYlGn_r", alpha=0.7)
    ax.contour(TH, THD, V_grid, levels=levels, colors="k", linewidths=0.5, alpha=0.5)
    plt.colorbar(cs, ax=ax, label="V(x)")

    # Overlay trajectories
    ics, labels = make_eval_ics(region_scale)
    colors = plt.cm.tab10(range(len(ics)))
    for ic, label, c in zip(ics, labels, colors):
        xs, us, us_nom, Vs, violations = simulate(
            policy_params, ic, p, hidden_sizes, lqr_K, horizon, dt,
            policy_mode, lyap_params, lyap_cfg, clf_cfg, adapt_cfg)
        ls = "--" if label.startswith("OUT:") else "-"
        ax.plot(xs[:, 1], xs[:, 3], color=c, ls=ls, lw=1.5,
                label=label, alpha=0.8)
        ax.plot(xs[0, 1], xs[0, 3], 'o', color=c, ms=6)
    ax.plot(0, 0, 'k*', ms=12, zorder=10, label="goal")
    ax.set(xlabel="theta (rad)", ylabel="theta_dot (rad/s)",
           title="Phase portrait + V contours")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlim(-jnp.pi - 0.5, jnp.pi + 0.5)

    # --- Plot 2: V along trajectories ---
    ax2 = axes[1]
    ts = jnp.arange(horizon) * dt
    for ic, label, c in zip(ics, labels, colors):
        xs, us, us_nom, Vs, violations = simulate(
            policy_params, ic, p, hidden_sizes, lqr_K, horizon, dt,
            policy_mode, lyap_params, lyap_cfg, clf_cfg, adapt_cfg)
        ls = "--" if label.startswith("OUT:") else "-"
        ax2.plot(ts, Vs, color=c, ls=ls, lw=1.5, label=label)
    ax2.set(xlabel="time (s)", ylabel="V(x)", title="Lyapunov value along trajectories")
    ax2.set_yscale("symlog", linthresh=0.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=6)

    # --- Plot 3: Shield violation along trajectories ---
    ax3 = axes[2]
    for ic, label, c in zip(ics, labels, colors):
        xs, us, us_nom, Vs, violations = simulate(
            policy_params, ic, p, hidden_sizes, lqr_K, horizon, dt,
            policy_mode, lyap_params, lyap_cfg, clf_cfg, adapt_cfg)
        ls = "--" if label.startswith("OUT:") else "-"
        ax3.plot(ts, violations, color=c, ls=ls, lw=1.5, label=label)
    ax3.axhline(0, color="k", ls=":", lw=0.8)
    ax3.set(xlabel="time (s)", ylabel="CLF violation (a^T u - b)",
            title="Shield constraint violation (>0 = infeasible)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=6)

    mode_str = "learned P" if learn_lyap else "fixed P_lqr"
    fig.suptitle(f"{run_name}: CLF analysis ({mode_str})", fontsize=13)
    fig.tight_layout()
    out = os.path.join(run_dir, "lyapunov_analysis.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved Lyapunov analysis -> {out}")


def plot_comparison(runs_info):
    """Side-by-side comparison of all runs for the same eval ICs."""
    p = CartPoleParams()
    hidden_sizes = (64, 64)
    Q_lqr = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R_lqr = jnp.array([[0.01]])
    P_lqr, lqr_K = solve_cartpole_lqr(p, Q_lqr, R_lqr)
    dt = 0.02
    horizon = 400

    # Use ICs for region_scale=0.3 (common across all)
    ics, labels = make_eval_ics(0.3)
    # Just use the 3 inside-region ICs for comparison
    ics = ics[:3]
    labels = labels[:3]

    n_runs = len(runs_info)
    fig, axes = plt.subplots(len(ics), n_runs, figsize=(5 * n_runs, 4 * len(ics)),
                             squeeze=False)
    ts = jnp.arange(horizon) * dt

    for col, (run_dir, run_name, save_data, run_config) in enumerate(runs_info):
        policy_params = save_data["nn"]
        policy_mode = save_data["policy_mode"]
        clf_enabled = save_data.get("clf_enabled", False)
        lambda_clf = run_config.get("lambda_clf", 0.5)
        learn_lyap = run_config.get("learn_lyap", False)

        if clf_enabled and "lyap_params" in save_data:
            lyap_cfg = LyapunovConfig(
                mode="quadratic_learned" if learn_lyap else "quadratic_fixed",
                state_dim=4, x_eq=(0.0, 0.0, 0.0, 0.0), P_init=P_lqr)
            lyap_params = save_data["lyap_params"]
            clf_cfg = CLFConfig(enabled=True, lambda_clf=lambda_clf)
        else:
            lyap_cfg = None
            lyap_params = None
            clf_cfg = CLFConfig(enabled=False)
        adapt_cfg = AdaptiveConfig(adapt_enabled=False)

        for row, (ic, label) in enumerate(zip(ics, labels)):
            ax = axes[row, col]
            xs, us, us_nom, Vs, violations = simulate(
                policy_params, ic, p, hidden_sizes, lqr_K, horizon, dt,
                policy_mode, lyap_params, lyap_cfg, clf_cfg, adapt_cfg)

            ax.plot(ts, xs[:, 1], 'b-', lw=2, label="theta")
            ax.plot(ts, xs[:, 0], 'r--', lw=1, alpha=0.6, label="x_cart")
            ax.axhline(0, color="k", ls=":", lw=0.5)
            ax.set_ylim(-jnp.pi - 0.5, jnp.pi + 0.5)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(run_name, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\ntheta (rad)", fontsize=9)
            if row == len(ics) - 1:
                ax.set_xlabel("time (s)")
            ax.legend(fontsize=7, loc="upper right")

            # Print final theta
            final_theta = float(xs[-1, 1])
            ax.text(0.02, 0.02, f"final: {final_theta:.3f} rad",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.suptitle("Comparison: in-region ICs across methods", fontsize=14)
    fig.tight_layout()
    out = "runs/comparison_cartpole.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved comparison -> {out}")


def main():
    run_dirs = [
        ("runs/step1_nn_only_r03", "Step1: NN-only"),
        ("runs/step2a_clf_fixed_r03", "Step2a: CLF fixed"),
        ("runs/step2b_clf_learned_r03", "Step2b: CLF learned"),
    ]

    runs_info = []
    for run_dir, run_name in run_dirs:
        if not os.path.exists(os.path.join(run_dir, "policy_params.pkl")):
            print(f"Skipping {run_name} (no saved params)")
            continue
        print(f"\n{'='*60}")
        print(f"  {run_name} ({run_dir})")
        print(f"{'='*60}")
        save_data, run_config = load_run(run_dir)
        print(f"  mode={save_data['policy_mode']}, clf={save_data.get('clf_enabled')}, "
              f"region={run_config.get('region_scale')}")

        # Rerun eval with updated ICs
        plot_eval(run_dir, run_name, save_data, run_config)

        # Lyapunov/shield visualizations (CLF runs only)
        plot_lyapunov_and_shield(run_dir, run_name, save_data, run_config)

        runs_info.append((run_dir, run_name, save_data, run_config))

    # Cross-run comparison
    if len(runs_info) >= 2:
        print(f"\n{'='*60}")
        print(f"  Cross-run comparison")
        print(f"{'='*60}")
        plot_comparison(runs_info)

    print("\nDone!")


if __name__ == "__main__":
    main()
