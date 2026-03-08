"""Evaluate a trained swing-up policy: simulate, plot, and animate.

Usage (from repo root):
    python scripts/eval_swingup.py                         # default runs/swingup
    python scripts/eval_swingup.py --load-dir runs/swingup --animate
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from adaptive_clf.configs import (
    AcrobotParams,
    AdaptiveConfig,
    AdaptiveState,
    CLFConfig,
    LyapunovConfig,
    RolloutConfig,
)
from adaptive_clf.acrobot import (
    acrobot_dynamics_true,
    rk4_step,
    solve_lqr_P,
)
from adaptive_clf.nn import policy_apply
from adaptive_clf.rollout import default_experiment_setup, sample_uniform_batch
from adaptive_clf.adaptive import init_adaptive_state, make_policy_observation


def simulate_policy(policy_params, lyap_params, lyap_cfg, x0, p, dt, n_steps,
                    a_true=0.0, K_lqr=None, P_lqr_blend=None,
                    lqr_V_threshold=5.0, lqr_temperature=1.0):
    """Simulate a trained policy from x0, returning trajectories."""
    from adaptive_clf.rollout import lqr_blend_control

    adapt_cfg = AdaptiveConfig()
    adaptive_state = init_adaptive_state(p, adapt_cfg)
    hidden_sizes = (64, 64)
    a_true_arr = jnp.array(a_true)

    xs, us = [np.array(x0)], []
    x = x0
    for _ in range(n_steps):
        obs = make_policy_observation(x, adaptive_state, adapt_cfg)
        u = policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes)
        u_clipped = jnp.clip(u, p.u_min, p.u_max)

        if K_lqr is not None and P_lqr_blend is not None:
            u_clipped = lqr_blend_control(
                u_clipped, x, K_lqr, P_lqr_blend,
                V_threshold=lqr_V_threshold,
                temperature=lqr_temperature,
                u_min=p.u_min, u_max=p.u_max,
            )

        us.append(float(u_clipped))
        x = rk4_step(acrobot_dynamics_true, x, u_clipped, a_true_arr, dt, p)
        xs.append(np.array(x))
    return np.array(xs), np.array(us)


def plot_evaluation(xs, us, dt, save_path):
    """Plot state trajectories and control input."""
    n = len(xs)
    t = np.arange(n) * dt
    t_u = t[:-1]

    q1_abs = np.pi + xs[:, 0]
    q2_abs = xs[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, np.degrees(q1_abs), label="$q_1$ (abs)")
    axes[0, 0].plot(t, np.degrees(q2_abs), "--", label="$q_2$ (abs)")
    axes[0, 0].axhline(180, color="gray", ls=":", lw=0.8, label="upright")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("degrees")
    axes[0, 0].set_title("Joint angles (absolute)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, xs[:, 2], label="$\\dot{q}_1$")
    axes[0, 1].plot(t, xs[:, 3], "--", label="$\\dot{q}_2$")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("rad/s")
    axes[0, 1].set_title("Joint velocities")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_u, us, color="tab:red")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("torque (N-m)")
    axes[1, 0].set_title("Control input")
    axes[1, 0].grid(True, alpha=0.3)

    state_norm = np.linalg.norm(xs, axis=1)
    axes[1, 1].plot(t, state_norm, color="tab:green")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("$|x|$ (upright coords)")
    axes[1, 1].set_title("State norm (0 = upright)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Swing-up policy evaluation", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved evaluation plot to {save_path}")


def animate_policy(xs, p, dt, save_path):
    """Generate a GIF animation of the trained policy."""
    import matplotlib.animation as animation

    l2 = 2 * p.lc2
    arm_len = p.l1 + l2

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = arm_len + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Swing-up policy")
    ax.plot(0, 0, "ko", markersize=8, zorder=5)

    (link_line,) = ax.plot([], [], "o-", color="steelblue", lw=4,
                            markersize=8, markerfacecolor="white",
                            markeredgecolor="steelblue", markeredgewidth=2, zorder=4)
    (trail_line,) = ax.plot([], [], "-", color="coral", lw=1.5, alpha=0.5)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    trail_x, trail_y = [], []
    trail_len = 30

    def update(frame):
        dq1, dq2 = xs[frame, 0], xs[frame, 1]
        q1 = np.pi + dq1
        q2 = dq2
        elbow = np.array([p.l1 * np.sin(q1), -p.l1 * np.cos(q1)])
        tip = elbow + np.array([l2 * np.sin(q1 + q2), -l2 * np.cos(q1 + q2)])
        link_line.set_data([0, elbow[0], tip[0]], [0, elbow[1], tip[1]])

        trail_x.append(tip[0])
        trail_y.append(tip[1])
        if len(trail_x) > trail_len:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)
        time_text.set_text(f"t = {frame * dt:.2f} s")
        return link_line, trail_line, time_text

    interval = max(int(dt * 1000), 20)
    anim = animation.FuncAnimation(fig, update, frames=len(xs),
                                    interval=interval, blit=True)
    anim.save(save_path, writer="pillow", fps=int(1 / dt))
    plt.close(fig)
    print(f"Saved animation to {save_path}")


def plot_multi_ic(all_results, dt, save_path):
    """Plot state trajectories from multiple initial conditions on shared axes."""
    n_ics = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_ics))

    for i, (label, xs, us) in enumerate(all_results):
        t = np.arange(len(xs)) * dt
        t_u = t[:-1]
        color = cmap[i]
        q1_abs = np.pi + xs[:, 0]

        axes[0, 0].plot(t, np.degrees(q1_abs), color=color, label=label, alpha=0.8)
        axes[0, 1].plot(t, xs[:, 2], color=color, label=label, alpha=0.8)
        axes[1, 0].plot(t_u, us, color=color, label=label, alpha=0.8)
        axes[1, 1].plot(t, np.linalg.norm(xs, axis=1), color=color, label=label, alpha=0.8)

    axes[0, 0].axhline(180, color="gray", ls=":", lw=0.8)
    axes[0, 0].set_ylabel("degrees")
    axes[0, 0].set_title("$q_1$ (absolute)")

    axes[0, 1].set_ylabel("rad/s")
    axes[0, 1].set_title("$\\dot{q}_1$")

    axes[1, 0].set_ylabel("torque (N-m)")
    axes[1, 0].set_title("Control input")

    axes[1, 1].set_ylabel("$|x|$")
    axes[1, 1].set_title("State norm (0 = upright)")

    for ax in axes.flat:
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Policy evaluation from multiple initial conditions", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved multi-IC plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate swing-up policy")
    parser.add_argument("--load-dir", type=str, default="runs/swingup")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--friction", type=float, default=0.0)
    parser.add_argument("--multi-ic", action="store_true",
                        help="Test from multiple initial conditions")
    parser.add_argument("--sample-train-dist", type=int, default=0, metavar="N",
                        help="Sample N initial conditions from training distribution (uniform)")
    parser.add_argument("--eval-seed", type=int, default=123)
    args = parser.parse_args()

    p = AcrobotParams()
    dt = 0.02
    n_steps = int(args.duration / dt)

    params_path = os.path.join(args.load_dir, "policy_params.pkl")
    print(f"Loading policy from {params_path}")
    with open(params_path, "rb") as f:
        policy_params = pickle.load(f)

    policy_params = jax.tree.map(jnp.asarray, policy_params)

    Q_lqr = jnp.diag(jnp.array([40.0, 40.0, 8.0, 8.0]))
    R_lqr = jnp.array([[0.5]])
    P_lqr, K_lqr = solve_lqr_P(p, Q=Q_lqr, R=R_lqr, a_nom=0.0)
    key = jax.random.PRNGKey(42)
    _, lyap_params, lyap_cfg = default_experiment_setup(key, p, P_lqr)

    run_config_path = os.path.join(args.load_dir, "run_config.pkl")
    blend_kwargs = {}
    run_config = {}
    if os.path.exists(run_config_path):
        with open(run_config_path, "rb") as f:
            run_config = pickle.load(f)
        if run_config.get("lqr_blend", False):
            K_saved = run_config.get("K_lqr", K_lqr)
            P_saved = run_config.get("P_lqr", P_lqr)
            blend_kwargs = {
                "K_lqr": jnp.asarray(K_saved),
                "P_lqr_blend": jnp.asarray(P_saved),
                "lqr_V_threshold": run_config.get("lqr_V_threshold", 5.0),
                "lqr_temperature": run_config.get("lqr_temperature", 1.0),
            }
            print(f"LQR blend enabled (V_thresh={blend_kwargs['lqr_V_threshold']:.1f})")

    if args.sample_train_dist > 0:
        n_eval = args.sample_train_dist
        eval_key = jax.random.PRNGKey(args.eval_seed)
        if os.path.exists(run_config_path) and "dq1_range" in run_config:
            batch_x0, _ = sample_uniform_batch(
                eval_key, n_eval, p,
                dq1_range=run_config["dq1_range"],
                dq2_range=run_config["dq2_range"],
                w_range=run_config["w_range"],
            )
            print(f"Sampling {n_eval} ICs from training region "
                  f"(scale={run_config['region_scale']}), seed={args.eval_seed}")
        else:
            batch_x0, _ = sample_uniform_batch(eval_key, n_eval, p)
            print(f"Sampling {n_eval} ICs from full state space (no run_config), seed={args.eval_seed}")

        all_results = []
        for i in range(n_eval):
            x0 = batch_x0[i]
            label = (f"dq1={float(x0[0]):+.2f} dq2={float(x0[1]):+.2f} "
                     f"w1={float(x0[2]):+.1f} w2={float(x0[3]):+.1f}")
            xs, us = simulate_policy(policy_params, lyap_params, lyap_cfg,
                                     x0, p, dt, n_steps, a_true=args.friction,
                                     **blend_kwargs)
            final_norm = np.linalg.norm(xs[-1])
            print(f"  {label:45s}  |x_T| = {final_norm:.4f}")
            all_results.append((label, xs, us))

        save_path = os.path.join(args.load_dir, "eval_train_dist.png")
        plot_multi_ic(all_results, dt, save_path)
    elif args.multi_ic:
        ic_list = [
            ("upright +0.1",   jnp.array([0.1, 0.0, 0.0, 0.0])),
            ("upright +0.3",   jnp.array([0.3, 0.0, 0.0, 0.0])),
            ("upright +0.5",   jnp.array([0.5, 0.0, 0.0, 0.0])),
            ("upright +1.0",   jnp.array([1.0, 0.0, 0.0, 0.0])),
            ("horizontal",     jnp.array([-jnp.pi / 2, 0.0, 0.0, 0.0])),
            ("3/4 down",       jnp.array([-3 * jnp.pi / 4, 0.0, 0.0, 0.0])),
            ("hang-down",      jnp.array([-jnp.pi, 0.0, 0.0, 0.0])),
            ("hang-down +vel", jnp.array([-jnp.pi, 0.0, 1.0, 0.0])),
        ]
        all_results = []
        for label, x0 in ic_list:
            xs, us = simulate_policy(policy_params, lyap_params, lyap_cfg,
                                     x0, p, dt, n_steps, a_true=args.friction,
                                     **blend_kwargs)
            final_norm = np.linalg.norm(xs[-1])
            print(f"  {label:20s}  |x_T| = {final_norm:.4f}")
            all_results.append((label, xs, us))

        save_path = os.path.join(args.load_dir, "eval_multi_ic.png")
        plot_multi_ic(all_results, dt, save_path)
    else:
        x0 = jnp.array([-jnp.pi, 0.0, 0.0, 0.0], dtype=jnp.float32)
        print(f"Simulating from hang-down for {args.duration}s, a={args.friction}")

        xs, us = simulate_policy(policy_params, lyap_params, lyap_cfg, x0, p, dt, n_steps,
                                 a_true=args.friction, **blend_kwargs)

        final_norm = np.linalg.norm(xs[-1])
        print(f"Final state: {xs[-1]}")
        print(f"Final |x| = {final_norm:.4f} (0 = upright)")

        plot_path = os.path.join(args.load_dir, "eval_swingup.png")
        plot_evaluation(xs, us, dt, plot_path)

        if args.animate:
            anim_path = os.path.join(args.load_dir, "eval_swingup.gif")
            animate_policy(xs, p, dt, anim_path)


if __name__ == "__main__":
    main()
