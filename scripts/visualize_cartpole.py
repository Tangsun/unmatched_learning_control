"""Comprehensive visualization of the trained hybrid LQR+NN cart-pole controller.

Generates a multi-panel figure showing:
  1. Training curves (loss + gradient norm)
  2. Trajectory rollouts from diverse initial conditions
  3. Phase portrait (theta vs theta_dot)
  4. Policy blending weight alpha(theta)
  5. NN vs LQR control decomposition
  6. Settle-time bar chart across test cases
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_clf.cartpole import CartPoleParams, rk4_step_cartpole
from adaptive_clf.nn import policy_apply

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "swingup_hybrid_v1"
PARAMS_PATH = RUN_DIR / "policy_params.pkl"
HISTORY_PATH = RUN_DIR / "history.pkl"


def load_policy():
    with open(PARAMS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["nn"], data["lqr_K"]


def load_history():
    with open(HISTORY_PATH, "rb") as f:
        return pickle.load(f)


def wrap_angle(theta):
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def make_obs(x):
    return jnp.array([x[0], jnp.sin(x[1]), jnp.cos(x[1]) - 1.0, x[2], x[3]])


def hybrid_policy_components(nn_params, x, lqr_K, p, hidden_sizes):
    """Return (u_total, u_nn, u_lqr, alpha)."""
    obs = make_obs(x)
    u_nn = policy_apply(nn_params, obs, p.u_min, p.u_max, hidden_sizes)
    u_lqr = jnp.clip((-lqr_K @ x).squeeze(), p.u_min, p.u_max)
    alpha = jax.nn.sigmoid(8.0 * (jnp.cos(x[1]) - 0.5))
    u_total = alpha * u_lqr + (1.0 - alpha) * u_nn
    return u_total, u_nn, u_lqr, alpha


def simulate(nn_params, lqr_K, x0, p, hidden_sizes, steps=500, dt=0.02):
    xs, us, u_nns, u_lqrs, alphas = [], [], [], [], []
    x = x0
    for _ in range(steps):
        x = x.at[1].set(wrap_angle(x[1]))
        x = jnp.clip(x, -20.0, 20.0)
        u_total, u_nn, u_lqr, alpha = hybrid_policy_components(
            nn_params, x, lqr_K, p, hidden_sizes
        )
        xs.append(x)
        us.append(u_total)
        u_nns.append(u_nn)
        u_lqrs.append(u_lqr)
        alphas.append(alpha)
        x = rk4_step_cartpole(x, u_total, jnp.array(0.0), dt, p)
    x = x.at[1].set(wrap_angle(x[1]))
    xs.append(x)
    return {
        "x": jnp.stack(xs),
        "u": jnp.stack(us),
        "u_nn": jnp.stack(u_nns),
        "u_lqr": jnp.stack(u_lqrs),
        "alpha": jnp.stack(alphas),
    }


def settle_time(theta_deg, threshold=10.0, dt=0.02):
    """Time at which |theta| first drops below threshold and stays there."""
    for i in range(len(theta_deg)):
        if all(abs(theta_deg[j]) < threshold for j in range(i, min(i + 50, len(theta_deg)))):
            return i * dt
    return len(theta_deg) * dt


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    nn_params, lqr_K = load_policy()
    history = load_history()
    p = CartPoleParams()
    hs = (64, 64)
    dt = 0.02
    steps = 500
    ts = np.arange(steps) * dt

    test_cases = {
        "Balance θ=0.2":         jnp.array([0.0, 0.2, 0.0, 0.0]),
        "Moderate θ=1.5":        jnp.array([0.0, 1.5, 0.0, 0.0]),
        "Near-hang θ=2.8":       jnp.array([0.0, 2.8, 0.0, 0.0]),
        "Swing-up θ=π":          jnp.array([0.0, jnp.pi, 0.0, 0.0]),
        "Swing-up θ=−π":         jnp.array([0.0, -jnp.pi, 0.0, 0.0]),
        "Hard: x=1,θ=π,v=0.5":   jnp.array([1.0, jnp.pi, 0.5, 0.5]),
    }

    results = {}
    for name, x0 in test_cases.items():
        results[name] = simulate(nn_params, lqr_K, x0, p, hs, steps, dt)

    # ── colours ──
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(test_cases)))

    # ══════════════════════════════════════════════════════════════════
    # Figure 1: Training curves
    # ══════════════════════════════════════════════════════════════════
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 4.5))

    step_nums = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    grad_norms = [h["grad_norm"] for h in history]

    ax1a.plot(step_nums, losses, lw=0.6, color="#2b6cb0")
    window = min(50, len(losses) // 5)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax1a.plot(step_nums[window - 1:], smoothed, lw=2, color="#e53e3e",
                  label=f"moving avg (w={window})")
    ax1a.set_yscale("symlog", linthresh=1.0)
    ax1a.set_xlabel("Training step", fontsize=11)
    ax1a.set_ylabel("Loss", fontsize=11)
    ax1a.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)

    ax1b.plot(step_nums, grad_norms, lw=0.6, color="#38a169")
    if window > 1:
        sm_g = np.convolve(grad_norms, np.ones(window) / window, mode="valid")
        ax1b.plot(step_nums[window - 1:], sm_g, lw=2, color="#d69e2e",
                  label=f"moving avg (w={window})")
    ax1b.set_yscale("symlog", linthresh=1.0)
    ax1b.set_xlabel("Training step", fontsize=11)
    ax1b.set_ylabel("Gradient norm", fontsize=11)
    ax1b.set_title("Gradient Norm", fontsize=13, fontweight="bold")
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    fig1.suptitle("Training Curves — Hybrid LQR+NN Swing-up Policy", fontsize=14, fontweight="bold", y=1.01)
    fig1.tight_layout()
    fig1.savefig(RUN_DIR / "vis_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved → {RUN_DIR / 'vis_training.png'}")

    # ══════════════════════════════════════════════════════════════════
    # Figure 2: Trajectory rollouts (4 subplots)
    # ══════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
    ax_x, ax_th, ax_v, ax_u = axes2.flat

    for (name, res), col in zip(results.items(), palette):
        xs = np.array(res["x"])
        us = np.array(res["u"])
        ax_x.plot(ts, xs[:steps, 0], label=name, color=col, lw=1.4)
        ax_th.plot(ts, np.degrees(xs[:steps, 1]), label=name, color=col, lw=1.4)
        ax_v.plot(ts, xs[:steps, 2], color=col, lw=1.2, label=name)
        ax_v.plot(ts, xs[:steps, 3], color=col, lw=1.0, ls="--", alpha=0.6)
        ax_u.plot(ts, us[:steps], label=name, color=col, lw=1.2)

    ax_x.set_ylabel("x_cart (m)")
    ax_x.set_title("Cart Position", fontweight="bold")
    ax_th.set_ylabel("θ (deg)")
    ax_th.set_title("Pole Angle", fontweight="bold")
    ax_th.axhline(0, color="k", ls=":", lw=0.8)
    ax_th.axhspan(-10, 10, color="green", alpha=0.07, label="±10° zone")
    ax_v.set_ylabel("Velocity")
    ax_v.set_title("Velocities (solid=cart, dashed=pole)", fontweight="bold")
    ax_u.set_ylabel("Force (N)")
    ax_u.set_title("Control Input", fontweight="bold")
    ax_u.axhline(p.u_max, color="r", ls=":", lw=0.8, alpha=0.5, label="u limits")
    ax_u.axhline(p.u_min, color="r", ls=":", lw=0.8, alpha=0.5)

    for ax in axes2.flat:
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")

    fig2.suptitle("Closed-Loop Trajectories — 10s Simulations", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(RUN_DIR / "vis_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {RUN_DIR / 'vis_trajectories.png'}")

    # ══════════════════════════════════════════════════════════════════
    # Figure 3: Phase portrait + blending weight + control decomposition
    # ══════════════════════════════════════════════════════════════════
    fig3 = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.35, wspace=0.35)

    # (a) Phase portrait: theta vs theta_dot
    ax3a = fig3.add_subplot(gs[0, 0])
    for (name, res), col in zip(results.items(), palette):
        xs = np.array(res["x"])
        ax3a.plot(np.degrees(xs[:steps, 1]), np.degrees(xs[:steps, 3]),
                  color=col, lw=1.0, label=name, alpha=0.85)
        ax3a.plot(np.degrees(xs[0, 1]), np.degrees(xs[0, 3]),
                  "o", color=col, ms=6, zorder=5)
        ax3a.plot(np.degrees(xs[steps - 1, 1]), np.degrees(xs[steps - 1, 3]),
                  "s", color=col, ms=5, zorder=5)
    ax3a.axhline(0, color="k", ls=":", lw=0.5)
    ax3a.axvline(0, color="k", ls=":", lw=0.5)
    ax3a.set_xlabel("θ (deg)")
    ax3a.set_ylabel("θ̇ (deg/s)")
    ax3a.set_title("Phase Portrait (○=start, □=end)", fontweight="bold")
    ax3a.legend(fontsize=6.5)
    ax3a.grid(True, alpha=0.3)

    # (b) Blending weight alpha over time
    ax3b = fig3.add_subplot(gs[0, 1])
    for (name, res), col in zip(results.items(), palette):
        ax3b.plot(ts, np.array(res["alpha"]), color=col, lw=1.4, label=name)
    ax3b.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax3b.set_xlabel("Time (s)")
    ax3b.set_ylabel("α (LQR weight)")
    ax3b.set_title("Policy Blending Weight α(t)", fontweight="bold")
    ax3b.set_ylim(-0.05, 1.05)
    ax3b.legend(fontsize=6.5)
    ax3b.grid(True, alpha=0.3)
    ax3b.annotate("α≈1 → LQR dominates", xy=(0.5, 0.95), fontsize=8,
                  color="#2b6cb0", ha="center", transform=ax3b.transAxes)
    ax3b.annotate("α≈0 → NN dominates", xy=(0.5, 0.05), fontsize=8,
                  color="#e53e3e", ha="center", transform=ax3b.transAxes)

    # (c) Alpha as function of theta (static curve)
    ax3c = fig3.add_subplot(gs[0, 2])
    theta_range = np.linspace(-np.pi, np.pi, 500)
    alpha_curve = 1.0 / (1.0 + np.exp(-8.0 * (np.cos(theta_range) - 0.5)))
    ax3c.plot(np.degrees(theta_range), alpha_curve, lw=2.5, color="#805ad5")
    ax3c.fill_between(np.degrees(theta_range), alpha_curve, alpha=0.15, color="#805ad5")
    ax3c.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax3c.axvline(0, color="green", ls=":", lw=1, label="upright (θ=0)")
    ax3c.axvline(180, color="red", ls=":", lw=1, alpha=0.5)
    ax3c.axvline(-180, color="red", ls=":", lw=1, alpha=0.5, label="hanging (θ=±π)")
    ax3c.set_xlabel("θ (deg)")
    ax3c.set_ylabel("α")
    ax3c.set_title("Blending Function α(θ)", fontweight="bold")
    ax3c.set_ylim(-0.05, 1.05)
    ax3c.legend(fontsize=8)
    ax3c.grid(True, alpha=0.3)

    # (d) NN vs LQR control for swing-up case
    swingup_key = "Swing-up θ=π"
    res_su = results[swingup_key]
    ax3d = fig3.add_subplot(gs[1, 0])
    ax3d.plot(ts, np.array(res_su["u_nn"])[:steps], lw=1.3, color="#e53e3e",
              label="u_NN", alpha=0.8)
    ax3d.plot(ts, np.array(res_su["u_lqr"])[:steps], lw=1.3, color="#2b6cb0",
              label="u_LQR", alpha=0.8)
    ax3d.plot(ts, np.array(res_su["u"])[:steps], lw=2.0, color="k",
              label="u_total", alpha=0.9)
    ax3d.set_xlabel("Time (s)")
    ax3d.set_ylabel("Force (N)")
    ax3d.set_title(f"Control Decomposition ({swingup_key})", fontweight="bold")
    ax3d.legend(fontsize=8)
    ax3d.grid(True, alpha=0.3)

    # (e) Control decomposition for hard case
    hard_key = "Hard: x=1,θ=π,v=0.5"
    res_hard = results[hard_key]
    ax3e = fig3.add_subplot(gs[1, 1])
    ax3e.plot(ts, np.array(res_hard["u_nn"])[:steps], lw=1.3, color="#e53e3e",
              label="u_NN", alpha=0.8)
    ax3e.plot(ts, np.array(res_hard["u_lqr"])[:steps], lw=1.3, color="#2b6cb0",
              label="u_LQR", alpha=0.8)
    ax3e.plot(ts, np.array(res_hard["u"])[:steps], lw=2.0, color="k",
              label="u_total", alpha=0.9)
    ax3e.set_xlabel("Time (s)")
    ax3e.set_ylabel("Force (N)")
    ax3e.set_title(f"Control Decomposition ({hard_key})", fontweight="bold")
    ax3e.legend(fontsize=8)
    ax3e.grid(True, alpha=0.3)

    # (f) Settle-time bar chart
    ax3f = fig3.add_subplot(gs[1, 2])
    names_list = list(results.keys())
    settle_times = []
    for name in names_list:
        xs = np.array(results[name]["x"])
        theta_deg = np.degrees(xs[:steps, 1])
        settle_times.append(settle_time(theta_deg, threshold=10.0, dt=dt))

    bars = ax3f.barh(range(len(names_list)), settle_times, color=palette, edgecolor="white", height=0.6)
    ax3f.set_yticks(range(len(names_list)))
    ax3f.set_yticklabels(names_list, fontsize=8)
    ax3f.set_xlabel("Settle time to ±10° (s)")
    ax3f.set_title("Settle Times", fontweight="bold")
    ax3f.set_xlim(0, max(settle_times) * 1.3 + 0.3)
    for bar, st in zip(bars, settle_times):
        ax3f.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                  f"{st:.2f}s", va="center", fontsize=9, fontweight="bold")
    ax3f.grid(True, alpha=0.3, axis="x")
    ax3f.invert_yaxis()

    fig3.suptitle("Hybrid LQR+NN Policy — Analysis & Decomposition",
                  fontsize=15, fontweight="bold")
    fig3.savefig(RUN_DIR / "vis_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved → {RUN_DIR / 'vis_analysis.png'}")

    print("\nDone. All visualizations saved to", RUN_DIR)


if __name__ == "__main__":
    main()
