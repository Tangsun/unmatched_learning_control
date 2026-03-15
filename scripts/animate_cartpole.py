"""Animate the cart-pole using saved policies.

Usage (from repo root):
    python scripts/animate_cartpole.py                           # all runs
    python scripts/animate_cartpole.py --run runs/step1_nn_only_r03
    python scripts/animate_cartpole.py --run runs/step2a_clf_fixed_r03 --ic 0.5

Saves GIFs to each run directory.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from adaptive_clf.configs import AdaptiveConfig, AdaptiveState, CLFConfig, LyapunovConfig
from adaptive_clf.nn import policy_apply
from adaptive_clf.cartpole import (
    CartPoleParams, cartpole_affine_terms, rk4_step_cartpole, solve_cartpole_lqr,
)
from adaptive_clf.lyapunov import lyapunov_value_and_grad
from adaptive_clf.shield import clf_shield


# ---------------------------------------------------------------------------
# Policy (matches train_cartpole.py exactly)
# ---------------------------------------------------------------------------

def _wrap_angle(theta):
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))

def _make_obs(x):
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
             policy_mode, lyap_params=None, lyap_cfg=None, clf_cfg=None):
    """Simulate and return numpy arrays."""
    a_true = jnp.array(0.0)
    use_clf = (policy_mode == "clf" and clf_cfg is not None
               and clf_cfg.enabled and lyap_params is not None)
    adaptive_state = AdaptiveState(
        a_hat=jnp.array(0.0), info=jnp.array(1e-6), radius=jnp.array(0.0))

    xs, us = [np.array(x0)], []
    x = x0
    for _ in range(horizon):
        x = jnp.clip(x, -20.0, 20.0)
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

        us.append(float(u))
        x = rk4_step_cartpole(x, u, a_true, dt, p)
        xs.append(np.array(x))

    return np.array(xs), np.array(us)


def make_animation(xs, us, p, dt, title, trail_length=20):
    """Create cart-pole animation with cart, pole, and control subplot."""
    cart_w, cart_h = 0.4, 0.2
    pole_len = p.l * 2  # full pole length (l is half-length in some conventions)

    fig, (ax_cart, ax_ctrl) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # Cart-pole axes
    x_range = max(np.max(np.abs(xs[:, 0])) + 1.5, 2.5)
    ax_cart.set_xlim(-x_range, x_range)
    ax_cart.set_ylim(-0.8, 1.5)
    ax_cart.set_aspect("equal")
    ax_cart.grid(True, alpha=0.3)
    ax_cart.set_title(title, fontsize=11)

    # Ground / track
    ax_cart.axhline(0, color="gray", lw=2, zorder=0)
    ax_cart.fill_between([-x_range, x_range], -0.8, 0, color="#f0f0f0", zorder=0)

    # Cart (rectangle)
    cart_patch = patches.Rectangle(
        (0, 0), cart_w, cart_h,
        facecolor="steelblue", edgecolor="navy", lw=2, zorder=3)
    ax_cart.add_patch(cart_patch)

    # Wheels
    wheel_r = 0.04
    wheel_l = plt.Circle((0, 0), wheel_r, color="gray", zorder=4)
    wheel_r_circ = plt.Circle((0, 0), wheel_r, color="gray", zorder=4)
    ax_cart.add_patch(wheel_l)
    ax_cart.add_patch(wheel_r_circ)

    # Pole
    (pole_line,) = ax_cart.plot([], [], "o-", color="firebrick", lw=5,
                                 markersize=8, markerfacecolor="gold",
                                 markeredgecolor="firebrick", markeredgewidth=2,
                                 zorder=5, solid_capstyle="round")

    # Tip trail
    (trail_line,) = ax_cart.plot([], [], "-", color="coral", lw=1.5, alpha=0.4)

    time_text = ax_cart.text(0.02, 0.95, "", transform=ax_cart.transAxes,
                              fontsize=10, verticalalignment="top",
                              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    theta_text = ax_cart.text(0.02, 0.82, "", transform=ax_cart.transAxes,
                               fontsize=9, verticalalignment="top",
                               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    # Control axes
    t_all = np.arange(len(us)) * dt
    ax_ctrl.set_xlim(0, t_all[-1] if len(t_all) > 0 else 1)
    u_absmax = max(np.max(np.abs(us)) * 1.1, 1.0)
    ax_ctrl.set_ylim(-u_absmax, u_absmax)
    ax_ctrl.set_xlabel("time (s)")
    ax_ctrl.set_ylabel("force (N)")
    ax_ctrl.set_title("Control input")
    ax_ctrl.grid(True, alpha=0.3)
    ax_ctrl.axhline(p.u_max, color="gray", ls="--", lw=0.8, label=f"u_max={p.u_max}")
    ax_ctrl.axhline(p.u_min, color="gray", ls="--", lw=0.8)
    (ctrl_line,) = ax_ctrl.plot([], [], color="tab:red", lw=1.5)
    ax_ctrl.legend(fontsize=8)

    fig.tight_layout()

    trail_x, trail_y = [], []

    def init():
        pole_line.set_data([], [])
        trail_line.set_data([], [])
        ctrl_line.set_data([], [])
        time_text.set_text("")
        theta_text.set_text("")
        return pole_line, trail_line, ctrl_line, time_text, theta_text, cart_patch, wheel_l, wheel_r_circ

    def update(frame):
        x_c = xs[frame, 0]
        theta = xs[frame, 1]

        # Cart position
        cart_patch.set_xy((x_c - cart_w / 2, -cart_h / 2))
        wheel_l.set_center((x_c - cart_w / 3, -cart_h / 2 - wheel_r))
        wheel_r_circ.set_center((x_c + cart_w / 3, -cart_h / 2 - wheel_r))

        # Pole: theta=0 is upright, positive counterclockwise
        # Pole tip relative to pivot (top of cart)
        pivot_y = cart_h / 2
        tip_x = x_c + pole_len * np.sin(theta)
        tip_y = pivot_y + pole_len * np.cos(theta)

        pole_line.set_data([x_c, tip_x], [pivot_y, tip_y])

        # Trail
        trail_x.append(tip_x)
        trail_y.append(tip_y)
        if len(trail_x) > trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        time_text.set_text(f"t = {frame * dt:.2f} s")
        theta_text.set_text(f"theta = {theta:.3f} rad\nx_cart = {x_c:.3f} m")

        if frame < len(us):
            ctrl_line.set_data(t_all[:frame + 1], us[:frame + 1])

        return pole_line, trail_line, ctrl_line, time_text, theta_text, cart_patch, wheel_l, wheel_r_circ

    n_frames = len(xs)
    interval = max(int(dt * 1000), 20)
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=interval, blit=True,
    )
    return fig, anim


def load_and_animate(run_dir, theta0, duration=8.0):
    """Load a saved policy and animate from given initial theta."""
    with open(os.path.join(run_dir, "policy_params.pkl"), "rb") as f:
        save_data = pickle.load(f)
    with open(os.path.join(run_dir, "run_config.pkl"), "rb") as f:
        run_config = pickle.load(f)

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
    learn_lyap = run_config.get("learn_lyap", False)
    dt = run_config.get("dt", 0.02)
    horizon = int(duration / dt)

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

    x0 = jnp.array([0.0, theta0, 0.0, 0.0])
    xs, us = simulate(policy_params, x0, p, hidden_sizes, lqr_K, horizon, dt,
                      policy_mode, lyap_params, lyap_cfg, clf_cfg)

    run_name = os.path.basename(run_dir)
    mode_labels = {"hybrid": "hybrid LQR+NN", "nn_only": "NN only", "clf": "NN+CLF"}
    title = f"{run_name}: {mode_labels.get(policy_mode, policy_mode)}, theta0={theta0:.2f} rad"

    fig, anim = make_animation(xs, us, p, dt, title)

    tag = f"theta{theta0:.2f}".replace(".", "p").replace("-", "m")
    out_path = os.path.join(run_dir, f"cartpole_{tag}.gif")
    print(f"  Saving {out_path} ({len(xs)} frames) ...")
    anim.save(out_path, writer="pillow", fps=int(1 / dt))
    plt.close(fig)
    print(f"  Done!")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="Single run directory (default: all step runs)")
    parser.add_argument("--ic", type=float, nargs="+", default=None,
                        help="Initial theta values in rad (default: [0.47, -0.94, pi])")
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    if args.run:
        run_dirs = [args.run]
    else:
        run_dirs = [
            "runs/step1_nn_only_r03",
            "runs/step2a_clf_fixed_r03",
            "runs/step2b_clf_learned_r03",
        ]

    if args.ic:
        thetas = args.ic
    else:
        # One inside region (small), one at region boundary, one outside
        thetas = [0.47, -0.94, float(jnp.pi)]

    for run_dir in run_dirs:
        if not os.path.exists(os.path.join(run_dir, "policy_params.pkl")):
            print(f"Skipping {run_dir} (no saved params)")
            continue
        print(f"\n{'='*60}")
        print(f"  {run_dir}")
        print(f"{'='*60}")
        for theta0 in thetas:
            load_and_animate(run_dir, theta0, args.duration)


if __name__ == "__main__":
    main()
