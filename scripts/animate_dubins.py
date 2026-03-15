"""Animate Dubins car path-following with saved policies.

Usage:
    python scripts/animate_dubins.py --run-dir runs/dubins_lyap_d4_rand
    python scripts/animate_dubins.py --run-dir runs/dubins_lyap_d4_rand --a-true 0.3
    python scripts/animate_dubins.py --run-dir runs/dubins_lyap_d4_rand --ic 1.0 1.5 1.0

Shows top-down view: reference path (x-axis), car position in world frame,
heading arrow, and control subplots.
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

from adaptive_clf.configs import AdaptiveConfig, AdaptiveState, CLFConfig
from adaptive_clf.integrator import rk4_step_generic
from adaptive_clf.lyapunov import lyapunov_value_and_grad
from adaptive_clf.nn import policy_apply
from adaptive_clf.shield import clf_shield
from adaptive_clf.systems import get_system
from adaptive_clf.adaptive import adaptive_update_generic


def _get_u_bounds(spec):
    if "u_min" in spec:
        return jnp.asarray(spec["u_min"]), jnp.asarray(spec["u_max"])
    p = spec["params"]
    return jnp.asarray(p.u_min), jnp.asarray(p.u_max)


def simulate_dubins(policy_params, lyap_params, lyap_cfg, x0, spec, data,
                    a_true, use_shield, horizon=300, dt=0.05):
    """Run rollout and return numpy arrays for animation."""
    p = spec["params"]
    wrap_fn = spec["wrap_state"]
    dynamics_fn = spec["dynamics_fn"]
    affine_fn = spec["affine_terms_fn"]
    make_obs = spec["make_obs"]
    u_lo, u_hi = _get_u_bounds(spec)

    hidden_sizes = data["hidden_sizes"]
    lambda_clf = data["lambda_clf"]
    out_dim = data.get("ctrl_dim", 2)
    use_adapt = data.get("use_adapt", False)
    adapt_cfg = data.get("adapt_cfg", AdaptiveConfig())

    clf_cfg = CLFConfig(enabled=True, lambda_clf=lambda_clf)

    if use_adapt and adapt_cfg.adapt_enabled:
        adapt_state = AdaptiveState(
            a_hat=jnp.array(0.0),
            info=jnp.asarray(adapt_cfg.info_init),
            radius=jnp.array(1.0))
    else:
        adapt_state = AdaptiveState(
            a_hat=jnp.array(0.0), info=jnp.array(1e-6), radius=jnp.array(0.0))

    a_true = jnp.asarray(a_true, dtype=jnp.float32)
    xs, us, Vs, a_hats = [np.array(x0)], [], [], []
    x = x0

    for _ in range(horizon):
        x = wrap_fn(x)
        obs = make_obs(x)

        if use_adapt and adapt_cfg.adapt_enabled:
            obs = jnp.concatenate([obs, jnp.array([adapt_state.a_hat, adapt_state.radius])])

        u_nom = policy_apply(policy_params, obs, u_lo, u_hi, hidden_sizes, out_dim=out_dim)

        if use_shield:
            u, _ = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=adapt_state,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_terms_fn=affine_fn,
                input_bounds=(u_lo, u_hi),
            )
            u = jnp.clip(u, u_lo, u_hi)
        else:
            u = jnp.clip(u_nom, u_lo, u_hi)

        V, _ = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)

        us.append(np.array(u))
        Vs.append(float(V))
        a_hats.append(float(adapt_state.a_hat))

        x = rk4_step_generic(dynamics_fn, x, u, a_true, dt, p)

        if use_adapt and adapt_cfg.adapt_enabled:
            adapt_state = adaptive_update_generic(
                adapt_state, x, u, a_true, dt, p, adapt_cfg,
                affine_terms_fn=affine_fn, dynamics_fn=dynamics_fn)

        xs.append(np.array(x))

    return np.array(xs), np.array(us), np.array(Vs), np.array(a_hats)


def error_to_world(e_x, e_y, e_theta, v_ref, t):
    """Convert error-frame coords to world frame.

    Reference path: x_ref = v_ref * t, y_ref = 0, theta_ref = 0.
    World coords: x_w = x_ref + e_x, y_w = e_y, theta_w = e_theta.
    """
    x_ref = v_ref * t
    x_w = x_ref + e_x
    y_w = e_y
    theta_w = e_theta
    return x_w, y_w, theta_w


def make_animation(xs, us, Vs, a_hats, a_true, dt, v_ref, title,
                   use_adapt=False, trail_length=60):
    """Create top-down Dubins car animation."""
    horizon = len(us)
    t_all = np.arange(horizon) * dt

    # Convert to world frame
    x_w_all = []
    y_w_all = []
    theta_w_all = []
    for i in range(len(xs)):
        xw, yw, tw = error_to_world(xs[i, 0], xs[i, 1], xs[i, 2], v_ref, i * dt)
        x_w_all.append(float(xw))
        y_w_all.append(float(yw))
        theta_w_all.append(float(tw))
    x_w_all = np.array(x_w_all)
    y_w_all = np.array(y_w_all)
    theta_w_all = np.array(theta_w_all)

    # Layout: car view (left), controls (right top), V (right bottom)
    n_right = 3 if use_adapt else 2
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(n_right, 2, width_ratios=[3, 2], hspace=0.4)
    ax_car = fig.add_subplot(gs[:, 0])
    ax_ctrl_v = fig.add_subplot(gs[0, 1])
    ax_ctrl_w = fig.add_subplot(gs[1, 1])
    if use_adapt:
        ax_adapt = fig.add_subplot(gs[2, 1])

    # Reference path extent
    x_ref_end = v_ref * horizon * dt
    pad = 2.0
    x_min = min(np.min(x_w_all) - pad, -pad)
    x_max = max(np.max(x_w_all) + pad, x_ref_end + pad)
    y_min = min(np.min(y_w_all) - pad, -3.0)
    y_max = max(np.max(y_w_all) + pad, 3.0)

    ax_car.set_xlim(x_min, x_max)
    ax_car.set_ylim(y_min, y_max)
    ax_car.set_aspect("equal")
    ax_car.grid(True, alpha=0.3)
    ax_car.set_xlabel("x (world)")
    ax_car.set_ylabel("y (world)")
    ax_car.set_title(title, fontsize=11)

    # Reference path
    ref_x = np.linspace(x_min, x_max, 100)
    ax_car.plot(ref_x, np.zeros_like(ref_x), "k--", lw=1.5, alpha=0.4, label="reference path")

    # Reference point marker
    (ref_pt,) = ax_car.plot([], [], "ko", markersize=6, alpha=0.5, label="ref point")

    # Trail
    (trail_line,) = ax_car.plot([], [], "-", color="dodgerblue", lw=1.5, alpha=0.5)

    # Car body (triangle)
    car_len = 0.3
    car_body = plt.Polygon([[0, 0]], closed=True, facecolor="steelblue",
                           edgecolor="navy", lw=2, zorder=5)
    ax_car.add_patch(car_body)

    # Heading arrow
    heading_arrow = ax_car.annotate("", xy=(0, 0), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle="->", color="firebrick", lw=2),
                                     zorder=6)

    # Side-slip arrow (shows a_true direction)
    slip_arrow = ax_car.annotate("", xy=(0, 0), xytext=(0, 0),
                                  arrowprops=dict(arrowstyle="->", color="green", lw=1.5, ls="--"),
                                  zorder=6)

    time_text = ax_car.text(0.02, 0.95, "", transform=ax_car.transAxes,
                            fontsize=10, verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    error_text = ax_car.text(0.02, 0.82, "", transform=ax_car.transAxes,
                             fontsize=9, verticalalignment="top",
                             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    ax_car.legend(fontsize=7, loc="lower right")

    # Control plots
    ax_ctrl_v.set_xlim(0, t_all[-1] if len(t_all) > 0 else 1)
    ax_ctrl_v.set_ylim(-0.1, 2.3)
    ax_ctrl_v.set_ylabel("v (speed)")
    ax_ctrl_v.axhline(v_ref, color="gray", ls=":", lw=0.8, label=f"v_ref={v_ref}")
    ax_ctrl_v.grid(True, alpha=0.3)
    ax_ctrl_v.legend(fontsize=7)
    (ctrl_v_line,) = ax_ctrl_v.plot([], [], color="tab:blue", lw=1.5)

    ax_ctrl_w.set_xlim(0, t_all[-1] if len(t_all) > 0 else 1)
    w_max = max(np.max(np.abs(us[:, 1])) * 1.1, 0.5)
    ax_ctrl_w.set_ylim(-w_max, w_max)
    ax_ctrl_w.set_ylabel("omega (turn rate)")
    ax_ctrl_w.set_xlabel("time (s)")
    ax_ctrl_w.axhline(0, color="k", lw=0.5)
    ax_ctrl_w.grid(True, alpha=0.3)
    (ctrl_w_line,) = ax_ctrl_w.plot([], [], color="tab:red", lw=1.5)

    if use_adapt:
        ax_adapt.set_xlim(0, t_all[-1] if len(t_all) > 0 else 1)
        ax_adapt.set_ylim(min(a_true - 0.2, -0.6), max(a_true + 0.2, 0.6))
        ax_adapt.set_ylabel("a_hat")
        ax_adapt.set_xlabel("time (s)")
        ax_adapt.axhline(a_true, color="r", ls="--", lw=1, label=f"a_true={a_true:.2f}")
        ax_adapt.grid(True, alpha=0.3)
        ax_adapt.legend(fontsize=7)
        (adapt_line,) = ax_adapt.plot([], [], color="tab:green", lw=1.5)

    try:
        fig.tight_layout()
    except Exception:
        pass

    trail_xs, trail_ys = [], []

    def _car_triangle(cx, cy, theta, size=0.3):
        """Return 3 vertices of a triangle pointing in heading direction."""
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Front tip
        front = [cx + size * cos_t, cy + size * sin_t]
        # Rear left
        rl = [cx - size * 0.5 * cos_t + size * 0.4 * sin_t,
              cy - size * 0.5 * sin_t - size * 0.4 * cos_t]
        # Rear right
        rr = [cx - size * 0.5 * cos_t - size * 0.4 * sin_t,
              cy - size * 0.5 * sin_t + size * 0.4 * cos_t]
        return [front, rl, rr]

    def init():
        trail_line.set_data([], [])
        ctrl_v_line.set_data([], [])
        ctrl_w_line.set_data([], [])
        time_text.set_text("")
        error_text.set_text("")
        artists = [trail_line, ctrl_v_line, ctrl_w_line, time_text, error_text,
                   car_body, ref_pt]
        if use_adapt:
            adapt_line.set_data([], [])
            artists.append(adapt_line)
        return artists

    def update(frame):
        cx = x_w_all[frame]
        cy = y_w_all[frame]
        theta = theta_w_all[frame]

        # Car triangle
        verts = _car_triangle(cx, cy, theta)
        car_body.set_xy(verts)

        # Heading arrow
        arr_len = 0.5
        heading_arrow.xy = (cx + arr_len * np.cos(theta), cy + arr_len * np.sin(theta))
        heading_arrow.set_position((cx, cy))

        # Side-slip arrow
        if abs(a_true) > 1e-3:
            slip_len = 0.3 * abs(a_true) / 0.5  # scale
            # Side-slip is perpendicular to heading (90 deg)
            slip_angle = theta + np.pi / 2 if a_true > 0 else theta - np.pi / 2
            slip_arrow.xy = (cx + slip_len * np.cos(slip_angle),
                            cy + slip_len * np.sin(slip_angle))
            slip_arrow.set_position((cx, cy))

        # Reference point
        x_ref = v_ref * frame * dt
        ref_pt.set_data([x_ref], [0])

        # Trail
        trail_xs.append(cx)
        trail_ys.append(cy)
        if len(trail_xs) > trail_length:
            trail_xs.pop(0)
            trail_ys.pop(0)
        trail_line.set_data(trail_xs, trail_ys)

        time_text.set_text(f"t = {frame * dt:.2f} s")
        error_text.set_text(
            f"e_x={xs[frame, 0]:.2f}  e_y={xs[frame, 1]:.2f}\n"
            f"e_th={xs[frame, 2]:.2f} rad  V={Vs[min(frame, len(Vs)-1)]:.3f}")

        if frame < len(us):
            ctrl_v_line.set_data(t_all[:frame + 1], us[:frame + 1, 0])
            ctrl_w_line.set_data(t_all[:frame + 1], us[:frame + 1, 1])
        if use_adapt and frame < len(a_hats):
            adapt_line.set_data(t_all[:frame + 1], a_hats[:frame + 1])

        artists = [trail_line, ctrl_v_line, ctrl_w_line, time_text, error_text,
                   car_body, ref_pt]
        if use_adapt:
            artists.append(adapt_line)
        return artists

    interval = max(int(dt * 1000), 20)
    anim = animation.FuncAnimation(
        fig, update, frames=len(xs), init_func=init,
        interval=interval, blit=True,
    )
    return fig, anim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ic", type=float, nargs=3, default=None,
                        help="Initial condition [e_x, e_y, e_theta]")
    parser.add_argument("--a-true", type=float, default=0.0)
    parser.add_argument("--shield", action="store_true", default=True)
    parser.add_argument("--no-shield", dest="shield", action="store_false")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=None,
                        help="Override duration in seconds (sets horizon)")
    args = parser.parse_args()

    if args.duration is not None:
        args.horizon = int(args.duration / args.dt)

    with open(os.path.join(args.run_dir, "policy_params.pkl"), "rb") as f:
        data = pickle.load(f)

    spec = get_system(data["system"])
    policy_params = data["nn"]
    lyap_params = data["lyap_params"]
    lyap_cfg = data["lyap_cfg"]
    v_ref = spec["params"].v_ref
    use_adapt = data.get("use_adapt", False)

    # Default ICs if none given
    if args.ic is not None:
        ic_list = [("custom", args.ic)]
    else:
        ic_list = [
            ("in_region", [0.5, 1.0, 0.8]),
            ("out_region", [3.0, 3.0, 2.5]),
        ]

    for ic_name, ic_vals in ic_list:
        x0 = jnp.array(ic_vals)
        print(f"Simulating IC={ic_vals}, a_true={args.a_true} ...")

        xs, us, Vs, a_hats = simulate_dubins(
            policy_params, lyap_params, lyap_cfg, x0, spec, data,
            a_true=args.a_true, use_shield=args.shield,
            horizon=args.horizon, dt=args.dt)

        run_name = os.path.basename(args.run_dir)
        a_str = f"a={args.a_true:.2f}"
        shield_str = "shield" if args.shield else "noshield"
        title = f"{run_name} | {a_str} | IC=({ic_vals[0]:.1f},{ic_vals[1]:.1f},{ic_vals[2]:.1f})"

        fig, anim = make_animation(
            xs, us, Vs, a_hats, args.a_true, args.dt, v_ref, title,
            use_adapt=use_adapt)

        a_tag = f"a{args.a_true:.2f}".replace(".", "p").replace("-", "m")
        out_path = os.path.join(
            args.run_dir, f"dubins_{ic_name}_{a_tag}_{shield_str}.gif")
        print(f"  Saving {out_path} ({len(xs)} frames) ...")
        anim.save(out_path, writer="pillow", fps=min(int(1 / args.dt), 30))
        plt.close(fig)
        print(f"  Done!")


if __name__ == "__main__":
    main()
