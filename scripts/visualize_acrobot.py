"""Animate the Acrobot as a two-link arm.

Usage (from repo root):
    python scripts/visualize_acrobot.py              # default: free-fall from near-upright
    python scripts/visualize_acrobot.py --mode lqr    # LQR balancing
    python scripts/visualize_acrobot.py --mode fall    # free-fall from horizontal

Saves  scripts/acrobot_<mode>.gif
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from adaptive_clf.configs import AcrobotParams
from adaptive_clf.acrobot import (
    acrobot_dynamics_true,
    rk4_step,
    solve_lqr_P,
)


def forward_kinematics(q1_abs, q2_abs, p: AcrobotParams):
    """Compute joint positions for plotting. Returns shoulder, elbow, tip."""
    l2 = 2 * p.lc2
    shoulder = np.array([0.0, 0.0])
    elbow = np.array([p.l1 * np.sin(q1_abs), -p.l1 * np.cos(q1_abs)])
    tip = elbow + np.array([l2 * np.sin(q1_abs + q2_abs), -l2 * np.cos(q1_abs + q2_abs)])
    return shoulder, elbow, tip


def simulate(x0, u_fn, a_true, p, dt, n_steps):
    xs = [np.array(x0)]
    us = []
    x = x0
    for _ in range(n_steps):
        u = u_fn(x)
        us.append(float(u))
        x = rk4_step(acrobot_dynamics_true, x, u, a_true, dt, p)
        xs.append(np.array(x))
    return np.array(xs), np.array(us)


def make_animation(xs, us, p, dt, title, trail_length=15):
    """Create a matplotlib animation of the acrobot."""
    l2 = 2 * p.lc2
    arm_len = p.l1 + l2
    margin = 0.3

    fig, (ax_arm, ax_ctrl) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # Arm axes
    lim = arm_len + margin
    ax_arm.set_xlim(-lim, lim)
    ax_arm.set_ylim(-lim, lim)
    ax_arm.set_aspect("equal")
    ax_arm.grid(True, alpha=0.3)
    ax_arm.set_title(title, fontsize=12)

    # Pivot marker
    ax_arm.plot(0, 0, "ko", markersize=8, zorder=5)

    (link_line,) = ax_arm.plot([], [], "o-", color="steelblue", lw=4,
                                markersize=8, markerfacecolor="white",
                                markeredgecolor="steelblue", markeredgewidth=2,
                                zorder=4)
    (trail_line,) = ax_arm.plot([], [], "-", color="coral", lw=1.5, alpha=0.5)
    time_text = ax_arm.text(0.02, 0.95, "", transform=ax_arm.transAxes,
                            fontsize=10, verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Control input axes
    t_all = np.arange(len(us)) * dt
    ax_ctrl.set_xlim(0, t_all[-1] if len(t_all) > 0 else 1)
    u_absmax = max(np.max(np.abs(us)) * 1.1, 1.0) if len(us) > 0 else 1.0
    ax_ctrl.set_ylim(-u_absmax, u_absmax)
    ax_ctrl.set_xlabel("time (s)")
    ax_ctrl.set_ylabel("torque (N-m)")
    ax_ctrl.set_title("Control input")
    ax_ctrl.grid(True, alpha=0.3)
    ax_ctrl.axhline(p.u_max, color="gray", ls="--", lw=0.8, label=f"u_max={p.u_max}")
    ax_ctrl.axhline(p.u_min, color="gray", ls="--", lw=0.8)
    (ctrl_line,) = ax_ctrl.plot([], [], color="tab:red", lw=1.5)
    ax_ctrl.legend(fontsize=8, loc="upper right")

    fig.tight_layout()

    trail_x, trail_y = [], []

    def init():
        link_line.set_data([], [])
        trail_line.set_data([], [])
        ctrl_line.set_data([], [])
        time_text.set_text("")
        return link_line, trail_line, ctrl_line, time_text

    def update(frame):
        dq1, dq2 = xs[frame, 0], xs[frame, 1]
        q1_abs = np.pi + dq1
        q2_abs = dq2
        shoulder, elbow, tip = forward_kinematics(q1_abs, q2_abs, p)

        link_line.set_data(
            [shoulder[0], elbow[0], tip[0]],
            [shoulder[1], elbow[1], tip[1]],
        )

        trail_x.append(tip[0])
        trail_y.append(tip[1])
        if len(trail_x) > trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        time_text.set_text(f"t = {frame * dt:.2f} s")

        if frame < len(us):
            ctrl_line.set_data(t_all[:frame + 1], us[:frame + 1])

        return link_line, trail_line, ctrl_line, time_text

    n_frames = len(xs)
    interval = max(int(dt * 1000), 20)  # ms per frame, at least 20ms
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=interval, blit=True,
    )
    return fig, anim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["upright", "lqr", "fall"], default="upright",
                        help="upright: small perturbation free-fall; "
                             "lqr: LQR stabilization; "
                             "fall: free-fall from horizontal")
    parser.add_argument("--friction", type=float, default=0.0, help="friction coefficient a")
    parser.add_argument("--duration", type=float, default=5.0, help="simulation time (s)")
    args = parser.parse_args()

    p = AcrobotParams()
    dt = 0.02
    n_steps = int(args.duration / dt)
    a_true = jnp.array(args.friction)

    if args.mode == "upright":
        x0 = jnp.array([0.3, 0.0, 0.0, 0.0], dtype=jnp.float32)
        u_fn = lambda x: jnp.array(0.0)
        title = f"Acrobot free-fall (x0=[0.3,0,0,0], a={args.friction})"

    elif args.mode == "lqr":
        x0 = jnp.array([0.01, 0.0, 0.0, 0.0], dtype=jnp.float32)
        Q = jnp.diag(jnp.array([40.0, 40.0, 8.0, 8.0]))
        R = jnp.array([[0.5]])
        _, K = solve_lqr_P(p, Q=Q, R=R, a_nom=args.friction)
        K_np = np.array(K).squeeze()
        u_fn = lambda x: jnp.clip(-K_np @ x, p.u_min, p.u_max)
        title = f"Acrobot LQR (x0=[0.01,0,0,0], a={args.friction})"

    elif args.mode == "fall":
        x0 = jnp.array([-np.pi / 2, 0.0, 0.0, 0.0], dtype=jnp.float32)
        u_fn = lambda x: jnp.array(0.0)
        title = f"Acrobot free-fall from horizontal (a={args.friction})"

    print(f"Simulating: {title}")
    xs, us = simulate(x0, u_fn, a_true, p, dt, n_steps)

    fig, anim = make_animation(xs, us, p, dt, title)

    out_path = os.path.join(os.path.dirname(__file__), f"acrobot_{args.mode}.gif")
    print(f"Saving animation to {out_path} ...")
    anim.save(out_path, writer="pillow", fps=int(1 / dt))
    print(f"Done! ({len(xs)} frames)")
    plt.close(fig)


if __name__ == "__main__":
    main()
