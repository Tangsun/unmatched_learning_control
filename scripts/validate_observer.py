"""Observer-based adaptation validation and animation.

Loads a saved dubins policy+Lyapunov and runs eval rollouts with the observer.
Produces:
  - 9-panel diagnostic plots per IC
  - Summary plot across all ICs
  - GIF animation showing car tracking + adaptation in real time

Usage:
    python scripts/validate_observer.py \
        --run-dir runs/dubins_observer_p2a_fast \
        --a-true 0.3 --observer-k 3.0 --observer-gamma 20.0

    # Skip animation (static plots only):
    python scripts/validate_observer.py --run-dir ... --no-animate
"""

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_clf.configs import (
    AdaptiveConfig, AdaptiveState, CLFConfig, LyapunovConfig,
)
from adaptive_clf.adaptive import (
    adaptive_update_observer, init_adaptive_state, make_adaptive_state,
)
from adaptive_clf.systems import get_system
from adaptive_clf.lyapunov import lyapunov_value_and_grad
from adaptive_clf.shield import clf_shield
from adaptive_clf.nn import policy_apply
from adaptive_clf.eval_rollout import make_eval_rollout_fn


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def make_rollout(policy_params, lyap_params, lyap_cfg, lambda_clf,
                 spec, hidden_sizes, adapt_cfg, horizon=400, dt=0.05,
                 use_shield=True, policy_obs_dim=None,
                 enforce_input_bounds=False, initial_radius=None):
    """Build a reusable compiled rollout function: fn(x0, a_true) -> dict."""
    return make_eval_rollout_fn(
        policy_params, lyap_params, lyap_cfg, lambda_clf,
        spec, hidden_sizes, adapt_cfg,
        horizon, dt, use_shield, policy_obs_dim,
        enforce_input_bounds=enforce_input_bounds,
        initial_radius=initial_radius,
    )


def run_eval_rollout(
    x0, a_true, policy_params, lyap_params, lyap_cfg, lambda_clf,
    spec, hidden_sizes, adapt_cfg, horizon=400, dt=0.05,
    use_shield=True, policy_obs_dim=None,
):
    """Single trajectory eval -- delegates to compiled lax.scan rollout."""
    fn = make_rollout(
        policy_params, lyap_params, lyap_cfg, lambda_clf,
        spec, hidden_sizes, adapt_cfg,
        horizon, dt, use_shield, policy_obs_dim,
    )
    return fn(x0, a_true)


def _run_eval_rollout_eager(
    x0, a_true, policy_params, lyap_params, lyap_cfg, lambda_clf,
    spec, hidden_sizes, adapt_cfg, horizon=400, dt=0.05,
    use_shield=True, policy_obs_dim=None,
    initial_radius=None,
):
    """Eager fallback for debugging (original Python-loop version)."""
    p = spec["params"]
    if policy_obs_dim is None:
        policy_obs_dim = spec["obs_dim"]
    affine_fn = spec["affine_terms_fn"]
    dynamics_fn = spec["dynamics_fn"]
    make_obs = spec["make_obs"]
    wrap_fn = spec["wrap_state"]
    state_dim = spec["state_dim"]
    ctrl_dim = spec["ctrl_dim"]

    clf_cfg = CLFConfig(enabled=use_shield, lambda_clf=lambda_clf, eps_proj=0.1)

    x = jnp.array(x0, dtype=jnp.float32)
    a_true_arr = jnp.asarray(a_true, dtype=jnp.float32)

    adapt_st = init_adaptive_state(
        p, adapt_cfg, state_dim=state_dim, x0=x, a_range=initial_radius,
    )

    # Storage
    xs, us, a_hats, radii = [np.array(x)], [], [], []
    etas, es, ws = [], [], []
    Vs, feasibles = [], []

    from adaptive_clf.integrator import rk4_step_generic

    for t in range(horizon):
        x = wrap_fn(x)

        # Build observation (only augment if policy was trained with adaptation)
        obs = make_obs(x)
        if policy_obs_dim > spec["obs_dim"]:
            obs = jnp.concatenate([obs, jnp.array([adapt_st.a_hat, adapt_st.radius])])

        # Policy
        if ctrl_dim == 1:
            u_nom = policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes, out_dim=1)
        else:
            u_min = spec["u_min"]
            u_max = spec["u_max"]
            u_nom = policy_apply(policy_params, obs, u_min, u_max, hidden_sizes, out_dim=ctrl_dim)

        # Shield
        if use_shield:
            u, shield_aux = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=adapt_st,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_terms_fn=affine_fn,
                alpha_max=0.0,
                input_bounds=(spec["u_min"], spec["u_max"]) if ctrl_dim > 1 else (p.u_min, p.u_max),
            )
            feasibles.append(float(shield_aux["feasible"]))
            V = shield_aux["V"]
        else:
            u = u_nom
            feasibles.append(1.0)
            V, _ = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)

        Vs.append(float(V))

        # Log observer state
        a_hats.append(float(adapt_st.a_hat))
        radii.append(float(adapt_st.radius))
        etas.append(float(jnp.linalg.norm(adapt_st.eta)))
        e = x - adapt_st.x_hat
        es.append(float(jnp.linalg.norm(e)))
        ws.append(float(jnp.linalg.norm(adapt_st.w)))
        us.append(np.array(u))

        # Observer update
        adapt_st = adaptive_update_observer(
            adapt_st, x, u, dt, p, adapt_cfg,
            affine_terms_fn=affine_fn,
            dynamics_fn=dynamics_fn,
            a_true=a_true_arr,
        )

        # True dynamics step (RK4)
        x = rk4_step_generic(dynamics_fn, x, u, a_true_arr, dt, p)
        xs.append(np.array(x))

    return {
        "xs": np.array(xs),
        "us": np.array(us),
        "a_hats": np.array(a_hats),
        "radii": np.array(radii),
        "etas": np.array(etas),
        "es": np.array(es),
        "ws": np.array(ws),
        "Vs": np.array(Vs),
        "feasibles": np.array(feasibles),
        "a_true": float(a_true),
    }


# ---------------------------------------------------------------------------
# Static diagnostic plots (unchanged)
# ---------------------------------------------------------------------------

def plot_observer_validation(results, save_path=None):
    """Plot observer convergence diagnostics."""
    t = np.arange(len(results["a_hats"]))
    a_true = results["a_true"]

    LgVs = results.get("LgVs")
    gradVs = results.get("gradVs")
    g_col_norms = results.get("g_col_norms")
    has_decomp = LgVs is not None and gradVs is not None
    nrows = 5 if has_decomp else (4 if LgVs is not None else 3)

    fig, axes = plt.subplots(nrows, 3, figsize=(16, 4 * nrows))

    ax = axes[0, 0]
    ax.plot(t, results["a_hats"], label="a_hat")
    ax.axhline(a_true, color="r", ls="--", label=f"a_true={a_true}")
    ax.fill_between(t,
                    results["a_hats"] - results["radii"],
                    results["a_hats"] + results["radii"],
                    alpha=0.2, color="blue", label="a_hat +/- radius")
    ax.set(xlabel="step", ylabel="a", title="Parameter estimate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, results["radii"], color="tab:orange")
    ax.plot(t, np.abs(np.array(results["a_hats"]) - a_true),
            color="tab:red", ls="--", label="|a_hat - a_true|")
    ax.set(xlabel="step", ylabel="radius", title="Radius vs true error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, results["etas"], color="tab:green")
    ax.set(xlabel="step", ylabel="||eta||", title="Auxiliary signal decay")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, results["es"], color="tab:purple")
    ax.set(xlabel="step", ylabel="||e||", title="Prediction error ||x - x_hat||")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, results["ws"], color="tab:brown")
    ax.set(xlabel="step", ylabel="||w||", title="Filter norm ||w||")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(t, results["Vs"], color="tab:blue")
    ax.set(xlabel="step", ylabel="V", title="Lyapunov V(x)")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    xs = results["xs"]
    ax.plot(xs[:, 0], xs[:, 1])
    ax.plot(xs[0, 0], xs[0, 1], "go", ms=8, label="start")
    ax.plot(xs[-1, 0], xs[-1, 1], "rs", ms=8, label="end")
    ax.set(xlabel="e_x", ylabel="e_y", title="Trajectory (error frame)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t, results["feasibles"], color="tab:green")
    ax.set(xlabel="step", ylabel="feasible", title="Shield feasibility")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    us = results["us"]
    ax.plot(t, us[:, 0], label="v")
    if us.shape[1] > 1:
        ax.plot(t, us[:, 1], label="omega")
    ax.set(xlabel="step", ylabel="u", title="Controls")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 3: LgV diagnostics
    if LgVs is not None:
        ax = axes[3, 0]
        LgV_norm = np.linalg.norm(LgVs, axis=-1)
        ax.plot(t, LgV_norm, color="tab:red")
        ax.axhline(0.1, color="gray", ls=":", lw=0.8, label="eps_proj=0.1")
        ax.set(xlabel="step", ylabel="||LgV||", title="||LgV|| (shield authority)")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[3, 1]
        for j in range(LgVs.shape[-1]):
            ax.plot(t, LgVs[:, j], label=f"LgV[{j}]", alpha=0.8)
        ax.set(xlabel="step", ylabel="LgV", title="LgV components")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[3, 2]
        ax.plot(t, LgV_norm**2, color="tab:red", label="||LgV||^2")
        ax.axhline(0.1, color="gray", ls=":", lw=0.8, label="eps_proj")
        ax.set(xlabel="step", ylabel="||LgV||^2",
               title="||LgV||^2 vs eps_proj (projection effective above)")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        for j in range(3):
            axes[3, j].set_visible(False)

    # Row 4: gradV / G(x) decomposition
    if has_decomp:
        ax = axes[4, 0]
        gradV_norm = np.linalg.norm(gradVs, axis=-1)
        ax.plot(t, gradV_norm, color="tab:blue")
        ax.set(xlabel="step", ylabel="||dV/dx||",
               title="||dV/dx|| (Lyapunov gradient norm)")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(True, alpha=0.3)

        ax = axes[4, 1]
        if g_col_norms is not None:
            for j in range(g_col_norms.shape[-1]):
                ax.plot(t, g_col_norms[:, j],
                        label=f"||g_{j}||", alpha=0.8)
        ax.set(xlabel="step", ylabel="||g_i(x)||",
               title="G(x) column norms (control effectiveness)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[4, 2]
        for j in range(gradVs.shape[-1]):
            ax.plot(t, gradVs[:, j], label=f"dV/dx[{j}]", alpha=0.7)
        ax.set(xlabel="step", ylabel="dV/dx_i",
               title="dV/dx components")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    elif nrows >= 5:
        for j in range(3):
            axes[4, j].set_visible(False)

    fig.suptitle(
        f"Observer Validation | a_true={a_true} | "
        f"final |a_hat-a_true|={abs(results['a_hats'][-1] - a_true):.4f} | "
        f"final radius={results['radii'][-1]:.4f}",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def _error_to_world(e_x, e_y, e_theta, v_ref, t):
    """Error-frame -> world-frame.  Reference: x_ref = v_ref*t along x-axis."""
    return v_ref * t + e_x, e_y, e_theta


def _car_triangle(cx, cy, theta, size=0.3):
    """Three vertices of an isoceles triangle pointing in heading direction."""
    ct, st = np.cos(theta), np.sin(theta)
    front = [cx + size * ct, cy + size * st]
    rl = [cx - size * 0.5 * ct + size * 0.4 * st,
          cy - size * 0.5 * st - size * 0.4 * ct]
    rr = [cx - size * 0.5 * ct - size * 0.4 * st,
          cy - size * 0.5 * st + size * 0.4 * ct]
    return [front, rl, rr]


def animate_observer(results, dt, v_ref, save_path, title="",
                     trail_length=80, fps=20, skip=2):
    """Create a GIF animation of the Dubins car with observer adaptation.

    Layout:
      Left  (large):   top-down car view with trail, heading, slip arrows
      Right (3 rows):  a_hat convergence, controls, Lyapunov V
    """
    xs = results["xs"]
    us = results["us"]
    Vs = np.array(results["Vs"])
    a_hats = np.array(results["a_hats"])
    radii = np.array(results["radii"])
    a_true = results["a_true"]
    horizon = len(us)
    t_all = np.arange(horizon) * dt

    # Pre-compute world-frame trajectory
    xw = np.zeros(len(xs))
    yw = np.zeros(len(xs))
    tw = np.zeros(len(xs))
    for i in range(len(xs)):
        xw[i], yw[i], tw[i] = _error_to_world(
            xs[i, 0], xs[i, 1], xs[i, 2], v_ref, i * dt)

    # ---- Figure layout ----
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 2], hspace=0.45, wspace=0.3)
    ax_car = fig.add_subplot(gs[:, 0])
    ax_adapt = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[1, 1])
    ax_lyap = fig.add_subplot(gs[2, 1])

    # ---- Car axis ----
    pad = 2.0
    x_min = min(np.min(xw) - pad, -pad)
    x_max = max(np.max(xw) + pad, v_ref * horizon * dt + pad)
    y_min = min(np.min(yw) - pad, -3.0)
    y_max = max(np.max(yw) + pad, 3.0)
    ax_car.set_xlim(x_min, x_max)
    ax_car.set_ylim(y_min, y_max)
    ax_car.set_aspect("equal")
    ax_car.grid(True, alpha=0.3)
    ax_car.set_xlabel("x (world)")
    ax_car.set_ylabel("y (world)")

    # Reference path
    ref_xs = np.linspace(x_min, x_max, 100)
    ax_car.plot(ref_xs, np.zeros_like(ref_xs), "k--", lw=1.5, alpha=0.4,
                label="reference path")
    (ref_pt,) = ax_car.plot([], [], "ko", ms=6, alpha=0.5)
    (trail_line,) = ax_car.plot([], [], "-", color="dodgerblue", lw=1.5, alpha=0.6)
    car_body = plt.Polygon([[0, 0]], closed=True, facecolor="steelblue",
                           edgecolor="navy", lw=2, zorder=5)
    ax_car.add_patch(car_body)
    heading_arrow = ax_car.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="firebrick", lw=2), zorder=6)
    slip_arrow = ax_car.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="green", lw=1.5, ls="--"), zorder=6)

    time_text = ax_car.text(
        0.02, 0.96, "", transform=ax_car.transAxes, fontsize=10,
        va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    state_text = ax_car.text(
        0.02, 0.80, "", transform=ax_car.transAxes, fontsize=9,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax_car.legend(fontsize=7, loc="lower right")

    # ---- Adaptation axis ----
    ax_adapt.set_xlim(0, t_all[-1])
    a_lo = min(np.min(a_hats - radii) - 0.1, a_true - 0.2, -0.1)
    a_hi = max(np.max(a_hats + radii) + 0.1, a_true + 0.2, 0.6)
    ax_adapt.set_ylim(a_lo, a_hi)
    ax_adapt.set_ylabel("a")
    ax_adapt.set_title("Parameter Adaptation", fontsize=10)
    ax_adapt.axhline(a_true, color="r", ls="--", lw=1.5, label=f"a_true={a_true:.2f}")
    ax_adapt.grid(True, alpha=0.3)
    ax_adapt.legend(fontsize=7, loc="lower right")
    (adapt_line,) = ax_adapt.plot([], [], color="tab:blue", lw=1.5, label="a_hat")
    adapt_fill = None  # will be updated each frame

    # ---- Controls axis ----
    ax_ctrl.set_xlim(0, t_all[-1])
    ax_ctrl.set_ylim(min(np.min(us) - 0.2, -2.5), max(np.max(us) + 0.2, 2.5))
    ax_ctrl.set_ylabel("control")
    ax_ctrl.set_title("Controls", fontsize=10)
    ax_ctrl.axhline(v_ref, color="gray", ls=":", lw=0.8)
    ax_ctrl.grid(True, alpha=0.3)
    (ctrl_v_line,) = ax_ctrl.plot([], [], color="tab:blue", lw=1.2, label="v")
    (ctrl_w_line,) = ax_ctrl.plot([], [], color="tab:orange", lw=1.2, label="omega")
    ax_ctrl.legend(fontsize=7, loc="upper right")

    # ---- Lyapunov axis ----
    ax_lyap.set_xlim(0, t_all[-1])
    ax_lyap.set_ylim(-0.02, max(np.max(Vs) * 1.1, 0.1))
    ax_lyap.set_ylabel("V(x)")
    ax_lyap.set_xlabel("time (s)")
    ax_lyap.set_title("Lyapunov Value", fontsize=10)
    ax_lyap.grid(True, alpha=0.3)
    (lyap_line,) = ax_lyap.plot([], [], color="tab:purple", lw=1.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ---- Animation state ----
    trail_xbuf, trail_ybuf = [], []

    # Frames to render (optionally skip for speed)
    frame_indices = list(range(0, horizon, skip))
    if frame_indices[-1] != horizon - 1:
        frame_indices.append(horizon - 1)

    def init():
        trail_line.set_data([], [])
        adapt_line.set_data([], [])
        ctrl_v_line.set_data([], [])
        ctrl_w_line.set_data([], [])
        lyap_line.set_data([], [])
        time_text.set_text("")
        state_text.set_text("")
        return []

    def update(idx):
        nonlocal adapt_fill
        frame = frame_indices[idx]
        cx, cy, theta = xw[frame], yw[frame], tw[frame]

        # Car triangle
        car_body.set_xy(_car_triangle(cx, cy, theta, size=0.3))

        # Heading arrow
        alen = 0.5
        heading_arrow.xy = (cx + alen * np.cos(theta), cy + alen * np.sin(theta))
        heading_arrow.set_position((cx, cy))

        # Side-slip arrow
        if abs(a_true) > 1e-3:
            slen = 0.4 * abs(a_true) / 0.5
            sa = theta + np.pi / 2 * np.sign(a_true)
            slip_arrow.xy = (cx + slen * np.cos(sa), cy + slen * np.sin(sa))
            slip_arrow.set_position((cx, cy))

        # Reference point
        ref_pt.set_data([v_ref * frame * dt], [0])

        # Trail
        trail_xbuf.append(cx)
        trail_ybuf.append(cy)
        if len(trail_xbuf) > trail_length:
            trail_xbuf.pop(0)
            trail_ybuf.pop(0)
        trail_line.set_data(trail_xbuf, trail_ybuf)

        # Info text
        time_text.set_text(f"t = {frame * dt:.2f} s")
        f_idx = min(frame, len(a_hats) - 1)
        state_text.set_text(
            f"e_x={xs[frame, 0]:+.3f}  e_y={xs[frame, 1]:+.3f}\n"
            f"e_th={xs[frame, 2]:+.3f} rad\n"
            f"a_hat={a_hats[f_idx]:.4f}  r={radii[f_idx]:.4f}\n"
            f"V={Vs[f_idx]:.4f}  |x|={np.linalg.norm(xs[frame]):.3f}")

        # Right panels: plot up to current time
        s = slice(0, frame + 1)
        adapt_line.set_data(t_all[s], a_hats[s])
        # Uncertainty band
        if adapt_fill is not None:
            adapt_fill.remove()
        adapt_fill = ax_adapt.fill_between(
            t_all[s], a_hats[s] - radii[s], a_hats[s] + radii[s],
            alpha=0.15, color="tab:blue")

        ctrl_v_line.set_data(t_all[s], us[s, 0])
        ctrl_w_line.set_data(t_all[s], us[s, 1])
        lyap_line.set_data(t_all[s], Vs[s])

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), init_func=init,
        interval=max(int(dt * skip * 1000), 30), blit=False)

    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved animation -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate observer-based adaptation")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to saved run (e.g. runs/dubins_shield_nobound_s2)")
    parser.add_argument("--a-true", type=float, default=0.3,
                        help="True uncertainty value")
    parser.add_argument("--observer-k", type=float, default=5.0)
    parser.add_argument("--observer-gamma", type=float, default=5.0)
    parser.add_argument("--observer-publish-mode", type=str, default=None,
                        choices=["nested", "aggressive"],
                        help="Override observer publication mode (default: use saved run config if available)")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-ics", type=int, default=6,
                        help="Number of initial conditions to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shield", action="store_true")
    parser.add_argument("--enforce-input-bounds", action="store_true",
                        help="Clip shield output to [u_min, u_max] after projection")
    parser.add_argument("--initial-radius", type=float, default=None,
                        help="Override initial uncertainty radius (default: from system params)")
    parser.add_argument("--animate", action="store_true",
                        help="Generate GIF animations (off by default)")
    parser.add_argument("--anim-skip", type=int, default=2,
                        help="Frame skip for animation (higher = faster)")
    parser.add_argument("--anim-fps", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    # Load saved model
    with open(os.path.join(args.run_dir, "policy_params.pkl"), "rb") as f:
        data = pickle.load(f)

    system = data.get("system", "dubins")
    spec = get_system(system)
    policy_params = data["nn"]
    lyap_params = data["lyap_params"]
    lyap_cfg = data["lyap_cfg"]
    lambda_clf = data.get("lambda_clf", 0.2)
    hidden_sizes = data.get("hidden_sizes", (64, 64))
    policy_obs_dim = data.get("obs_dim", spec["obs_dim"])
    v_ref = spec["params"].v_ref

    saved_adapt_cfg = data.get("adapt_cfg", None)
    observer_publish_mode = (
        args.observer_publish_mode
        if args.observer_publish_mode is not None
        else getattr(saved_adapt_cfg, "observer_publish_mode", "nested")
    )

    adapt_cfg = AdaptiveConfig(
        adapt_enabled=True,
        use_observer=True,
        observer_k=args.observer_k,
        observer_gamma=args.observer_gamma,
        observer_publish_mode=observer_publish_mode,
        stopgrad_obs=True,
    )

    save_dir = args.save_dir or os.path.join(args.run_dir, "observer_validation")
    os.makedirs(save_dir, exist_ok=True)

    # Sample initial conditions
    key = jax.random.PRNGKey(args.seed)
    x0s, _ = spec["sample_ics"](key, args.n_ics, 1.0, 0.0)

    print(f"Observer validation: system={system}, a_true={args.a_true}")
    print(f"  k={args.observer_k}, gamma={args.observer_gamma}, publish={observer_publish_mode}")
    print(f"  horizon={args.horizon}, dt={args.dt}")
    print(f"  shield={'OFF' if args.no_shield else 'ON'}"
          f"{' (input bounds enforced)' if args.enforce_input_bounds else ''}")
    print(f"  animate={'ON' if args.animate else 'OFF'} (skip={args.anim_skip})")
    print(f"  Testing {args.n_ics} ICs from: {args.run_dir}")
    print()

    # Build compiled rollout once, reuse for all ICs
    eval_fn = make_rollout(
        policy_params=policy_params,
        lyap_params=lyap_params,
        lyap_cfg=lyap_cfg,
        lambda_clf=lambda_clf,
        spec=spec,
        hidden_sizes=hidden_sizes,
        adapt_cfg=adapt_cfg,
        horizon=args.horizon,
        dt=args.dt,
        use_shield=not args.no_shield,
        policy_obs_dim=policy_obs_dim,
        enforce_input_bounds=args.enforce_input_bounds,
        initial_radius=args.initial_radius,
    )

    all_results = []
    for i, x0 in enumerate(x0s):
        x0 = np.array(x0)
        res = eval_fn(x0, args.a_true)
        all_results.append(res)

        final_err = abs(res["a_hats"][-1] - args.a_true)
        final_r = res["radii"][-1]
        final_xnorm = np.linalg.norm(res["xs"][-1])
        feas_rate = np.mean(res["feasibles"])
        print(f"  IC {i}: x0={x0} -> |a_hat-a|={final_err:.4f}, "
              f"r={final_r:.4f}, |xT|={final_xnorm:.3f}, feas={feas_rate:.2f}")

        # Static diagnostic plot
        plot_observer_validation(
            res,
            save_path=os.path.join(save_dir, f"observer_ic{i}.png"),
        )

        # Animation
        if args.animate:
            run_name = os.path.basename(args.run_dir)
            a_str = f"a_true={args.a_true:.2f}"
            ic_str = f"({x0[0]:.1f}, {x0[1]:.1f}, {x0[2]:.1f})"
            title = f"{run_name} | {a_str} | IC={ic_str}"
            animate_observer(
                res, dt=args.dt, v_ref=v_ref,
                save_path=os.path.join(save_dir, f"observer_ic{i}.gif"),
                title=title,
                trail_length=80,
                fps=args.anim_fps,
                skip=args.anim_skip,
            )

    # Summary plot: overlay a_hat convergence for all ICs
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, res in enumerate(all_results):
        t = np.arange(len(res["a_hats"]))
        axes[0].plot(t, res["a_hats"], alpha=0.7, label=f"IC{i}")
        axes[1].plot(t, res["radii"], alpha=0.7)
        axes[2].plot(t, np.abs(np.array(res["a_hats"]) - args.a_true), alpha=0.7)

    axes[0].axhline(args.a_true, color="r", ls="--", lw=2)
    axes[0].set(xlabel="step", ylabel="a_hat", title="Parameter estimate (all ICs)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set(xlabel="step", ylabel="radius", title="Uncertainty radius")
    axes[1].grid(True, alpha=0.3)

    axes[2].set(xlabel="step", ylabel="|a_hat - a_true|", title="Estimation error")
    axes[2].set_yscale("symlog", linthresh=1e-4)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Observer Summary | a_true={args.a_true} | k={args.observer_k} | gamma={args.observer_gamma}")
    fig.tight_layout()
    summary_path = os.path.join(save_dir, "observer_summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
