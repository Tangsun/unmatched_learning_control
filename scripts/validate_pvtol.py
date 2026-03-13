"""PVTOL observer-based adaptation validation and comparison.

Loads saved PVTOL policies and runs eval rollouts with/without observer.
Produces:
  - 9-panel diagnostic plots per IC (state, observer, Lyapunov)
  - Comparison table across runs and wind values
  - PVTOL animation (vehicle with thrust vector + wind arrow)

Usage:
    # Single run evaluation:
    python scripts/validate_pvtol.py \
        --run-dir runs/pvtol_p3_observer_scratch \
        --a-true 1.0 --observer-k 3.0 --observer-gamma 20.0

    # Compare multiple runs at multiple wind values:
    python scripts/validate_pvtol.py --compare \
        --run-dirs runs/pvtol_p1_baseline runs/pvtol_p2_randA runs/pvtol_p3_observer_scratch \
        --a-values 0.0 1.0 2.0 3.0 \
        --observer-k 3.0 --observer-gamma 20.0
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
    adaptive_update_observer, init_adaptive_state,
)
from adaptive_clf.systems import get_system
from adaptive_clf.lyapunov import lyapunov_value_and_grad
from adaptive_clf.shield import clf_shield
from adaptive_clf.nn import policy_apply
from adaptive_clf.integrator import rk4_step_generic
from adaptive_clf.eval_rollout import make_eval_rollout_fn, make_batched_metrics_fn


# ---------------------------------------------------------------------------
# Rollout (generic, works for any system)
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

    use_observer = adapt_cfg is not None and adapt_cfg.use_observer
    if adapt_cfg is not None and adapt_cfg.adapt_enabled:
        adapt_st = init_adaptive_state(p, adapt_cfg, state_dim=state_dim, x0=x)
    else:
        adapt_st = AdaptiveState(
            a_hat=jnp.array(0.0), info=jnp.array(1e-6),
            radius=jnp.array(0.0),
            x_hat=jnp.zeros(state_dim),
            w=jnp.zeros(state_dim),
            eta=jnp.zeros(state_dim))

    xs, us, a_hats, radii = [np.array(x)], [], [], []
    etas, es, ws = [], [], []
    Vs, feasibles = [], []

    for t in range(horizon):
        x = wrap_fn(x)

        obs = make_obs(x)
        if policy_obs_dim > spec["obs_dim"]:
            obs = jnp.concatenate([obs, jnp.array([adapt_st.a_hat, adapt_st.radius])])

        if ctrl_dim == 1:
            u_nom = policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes, out_dim=1)
        else:
            u_min = spec["u_min"]
            u_max = spec["u_max"]
            u_nom = policy_apply(policy_params, obs, u_min, u_max, hidden_sizes, out_dim=ctrl_dim)

        if use_shield:
            u, shield_aux = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=adapt_st,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_terms_fn=affine_fn,
                alpha_max=0.0,
            )
            feasibles.append(float(shield_aux["feasible"]))
            V = shield_aux["V"]
        else:
            if ctrl_dim == 1:
                u = jnp.clip(u_nom, p.u_min, p.u_max)
            else:
                u = jnp.clip(u_nom, spec["u_min"], spec["u_max"])
            feasibles.append(1.0)
            V, _ = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)

        Vs.append(float(V))

        a_hats.append(float(adapt_st.a_hat))
        radii.append(float(adapt_st.radius))
        etas.append(float(jnp.linalg.norm(adapt_st.eta)))
        e = x - adapt_st.x_hat
        es.append(float(jnp.linalg.norm(e)))
        ws.append(float(jnp.linalg.norm(adapt_st.w)))
        us.append(np.array(u))

        if use_observer:
            adapt_st = adaptive_update_observer(
                adapt_st, x, u, dt, p, adapt_cfg,
                affine_terms_fn=affine_fn,
            )

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
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_pvtol_diagnostics(results, save_path=None, title_extra=""):
    """9-panel diagnostic plot for PVTOL."""
    t = np.arange(len(results["a_hats"]))
    a_true = results["a_true"]
    xs = results["xs"]
    us = results["us"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Row 0: Observer
    ax = axes[0, 0]
    ax.plot(t, results["a_hats"], label="a_hat")
    ax.axhline(a_true, color="r", ls="--", label=f"a_true={a_true}")
    ax.fill_between(t,
                    results["a_hats"] - results["radii"],
                    results["a_hats"] + results["radii"],
                    alpha=0.2, color="blue")
    ax.set(xlabel="step", ylabel="a", title="Wind estimate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, results["radii"], color="tab:orange", label="radius")
    ax.plot(t, np.abs(np.array(results["a_hats"]) - a_true),
            color="tab:red", ls="--", label="|a_hat - a_true|")
    ax.set(xlabel="step", ylabel="radius", title="Radius vs true error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, results["Vs"], color="tab:purple")
    ax.set(xlabel="step", ylabel="V", title="Lyapunov V(x)")
    ax.grid(True, alpha=0.3)

    # Row 1: States
    ax = axes[1, 0]
    ax.plot(xs[:, 0], xs[:, 1])
    ax.plot(xs[0, 0], xs[0, 1], "go", ms=8, label="start")
    ax.plot(xs[-1, 0], xs[-1, 1], "rs", ms=8, label="end")
    ax.plot(0, 0, "k+", ms=12, mew=2, label="target")
    ax.set(xlabel="px (m)", ylabel="py (m)", title="Position trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    time = t * 0.05  # approximate
    ax.plot(time, xs[:-1, 0], label="px")
    ax.plot(time, xs[:-1, 1], label="py")
    ax.plot(time, xs[:-1, 2], label="theta")
    ax.set(xlabel="time (s)", ylabel="state", title="Position & angle")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(time, xs[:-1, 3], label="vx")
    ax.plot(time, xs[:-1, 4], label="vy")
    ax.plot(time, xs[:-1, 5], label="thetadot")
    ax.set(xlabel="time (s)", ylabel="state", title="Velocities")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: Controls & feasibility
    ax = axes[2, 0]
    ax.plot(time, us[:, 0], label="T (thrust)")
    ax.axhline(9.81, color="gray", ls=":", lw=0.8, label="mg")
    ax.set(xlabel="time (s)", ylabel="T (N)", title="Thrust")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(time, us[:, 1], label="tau (torque)", color="tab:orange")
    ax.set(xlabel="time (s)", ylabel="tau (N*m)", title="Torque")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.plot(time, results["feasibles"], color="tab:green")
    ax.set(xlabel="time (s)", ylabel="feasible", title="Shield feasibility")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    final_err = abs(results["a_hats"][-1] - a_true)
    final_xnorm = np.linalg.norm(xs[-1])
    fig.suptitle(
        f"PVTOL Validation | a_wind={a_true} | "
        f"|a_hat-a|={final_err:.4f} | |xT|={final_xnorm:.3f} {title_extra}",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PVTOL Animation
# ---------------------------------------------------------------------------

def _vehicle_shape(cx, cy, theta, width=0.6, height=0.15):
    """Rotated rectangle for PVTOL body."""
    ct, st = np.cos(theta), np.sin(theta)
    hw, hh = width / 2, height / 2
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    R = np.array([[ct, -st], [st, ct]])
    rotated = corners @ R.T + np.array([cx, cy])
    return rotated


def animate_pvtol(results, dt, save_path, title="",
                  trail_length=80, fps=20, skip=2):
    """GIF animation of PVTOL hovering with wind + observer adaptation."""
    xs = results["xs"]
    us = results["us"]
    Vs = np.array(results["Vs"])
    a_hats = np.array(results["a_hats"])
    radii = np.array(results["radii"])
    a_true = results["a_true"]
    horizon = len(us)
    t_all = np.arange(horizon) * dt

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 2], hspace=0.45, wspace=0.3)
    ax_pos = fig.add_subplot(gs[:, 0])
    ax_adapt = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[1, 1])
    ax_lyap = fig.add_subplot(gs[2, 1])

    # Position axis
    pad = 1.0
    x_min = min(np.min(xs[:, 0]) - pad, -3.0)
    x_max = max(np.max(xs[:, 0]) + pad, 3.0)
    y_min = min(np.min(xs[:, 1]) - pad, -3.0)
    y_max = max(np.max(xs[:, 1]) + pad, 3.0)
    ax_pos.set_xlim(x_min, x_max)
    ax_pos.set_ylim(y_min, y_max)
    ax_pos.set_aspect("equal")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_xlabel("px (m)")
    ax_pos.set_ylabel("py (m)")
    ax_pos.plot(0, 0, "k+", ms=15, mew=2, label="target", zorder=3)

    (trail_line,) = ax_pos.plot([], [], "-", color="dodgerblue", lw=1.5, alpha=0.6)
    vehicle_body = plt.Polygon([[0, 0]], closed=True, facecolor="steelblue",
                               edgecolor="navy", lw=2, zorder=5)
    ax_pos.add_patch(vehicle_body)

    thrust_arrow = ax_pos.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="firebrick", lw=2.5), zorder=6)

    # Wind arrow (fixed)
    if abs(a_true) > 0.01:
        wind_scale = min(abs(a_true) / 3.0, 1.0) * 1.5
        wind_x = x_min + 0.5
        wind_y = y_max - 0.5
        ax_pos.annotate(
            "", xy=(wind_x + wind_scale * np.sign(a_true), wind_y),
            xytext=(wind_x, wind_y),
            arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5))
        ax_pos.text(wind_x, wind_y + 0.3, f"wind={a_true:.1f} m/s^2",
                    fontsize=9, color="green", ha="left")

    time_text = ax_pos.text(
        0.02, 0.96, "", transform=ax_pos.transAxes, fontsize=10,
        va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    state_text = ax_pos.text(
        0.02, 0.78, "", transform=ax_pos.transAxes, fontsize=9,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax_pos.legend(fontsize=7, loc="lower right")

    # Adaptation axis
    ax_adapt.set_xlim(0, t_all[-1])
    a_lo = min(np.min(a_hats - radii) - 0.2, a_true - 0.3, -0.2)
    a_hi = max(np.max(a_hats + radii) + 0.2, a_true + 0.3, 0.5)
    ax_adapt.set_ylim(a_lo, a_hi)
    ax_adapt.set_ylabel("a (m/s^2)")
    ax_adapt.set_title("Wind Estimation", fontsize=10)
    ax_adapt.axhline(a_true, color="r", ls="--", lw=1.5, label=f"a_true={a_true:.1f}")
    ax_adapt.grid(True, alpha=0.3)
    ax_adapt.legend(fontsize=7, loc="lower right")
    (adapt_line,) = ax_adapt.plot([], [], color="tab:blue", lw=1.5)
    adapt_fill = None

    # Controls axis
    ax_ctrl.set_xlim(0, t_all[-1])
    ax_ctrl.set_ylim(min(np.min(us) - 1, -3), max(np.max(us) + 1, 22))
    ax_ctrl.set_ylabel("control")
    ax_ctrl.set_title("Controls", fontsize=10)
    ax_ctrl.axhline(9.81, color="gray", ls=":", lw=0.8, label="mg")
    ax_ctrl.grid(True, alpha=0.3)
    (ctrl_T_line,) = ax_ctrl.plot([], [], color="tab:blue", lw=1.2, label="T")
    (ctrl_tau_line,) = ax_ctrl.plot([], [], color="tab:orange", lw=1.2, label="tau")
    ax_ctrl.legend(fontsize=7, loc="upper right")

    # Lyapunov axis
    ax_lyap.set_xlim(0, t_all[-1])
    ax_lyap.set_ylim(-0.02, max(np.max(Vs) * 1.1, 0.1))
    ax_lyap.set_ylabel("V(x)")
    ax_lyap.set_xlabel("time (s)")
    ax_lyap.set_title("Lyapunov Value", fontsize=10)
    ax_lyap.grid(True, alpha=0.3)
    (lyap_line,) = ax_lyap.plot([], [], color="tab:purple", lw=1.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    trail_xbuf, trail_ybuf = [], []
    frame_indices = list(range(0, horizon, skip))
    if frame_indices[-1] != horizon - 1:
        frame_indices.append(horizon - 1)

    def init():
        trail_line.set_data([], [])
        adapt_line.set_data([], [])
        ctrl_T_line.set_data([], [])
        ctrl_tau_line.set_data([], [])
        lyap_line.set_data([], [])
        time_text.set_text("")
        state_text.set_text("")
        return []

    def update(idx):
        nonlocal adapt_fill
        frame = frame_indices[idx]
        px, py, theta = xs[frame, 0], xs[frame, 1], xs[frame, 2]

        # Vehicle body (rotated rectangle)
        vehicle_body.set_xy(_vehicle_shape(px, py, theta))

        # Thrust arrow (along body z-axis)
        T = us[frame, 0] if frame < len(us) else 9.81
        t_scale = T / 9.81 * 0.8  # normalize
        thrust_dx = -t_scale * np.sin(theta)
        thrust_dy = t_scale * np.cos(theta)
        thrust_arrow.xy = (px + thrust_dx, py + thrust_dy)
        thrust_arrow.set_position((px, py))

        # Trail
        trail_xbuf.append(px)
        trail_ybuf.append(py)
        if len(trail_xbuf) > trail_length:
            trail_xbuf.pop(0)
            trail_ybuf.pop(0)
        trail_line.set_data(trail_xbuf, trail_ybuf)

        # Info text
        time_text.set_text(f"t = {frame * dt:.2f} s")
        f_idx = min(frame, len(a_hats) - 1)
        state_text.set_text(
            f"px={px:+.3f}  py={py:+.3f}\n"
            f"th={theta:+.3f} rad\n"
            f"a_hat={a_hats[f_idx]:.3f}  r={radii[f_idx]:.3f}\n"
            f"V={Vs[f_idx]:.4f}  |x|={np.linalg.norm(xs[frame]):.3f}")

        # Right panels
        s = slice(0, frame + 1)
        adapt_line.set_data(t_all[s], a_hats[s])
        if adapt_fill is not None:
            adapt_fill.remove()
        adapt_fill = ax_adapt.fill_between(
            t_all[s], a_hats[s] - radii[s], a_hats[s] + radii[s],
            alpha=0.15, color="tab:blue")

        ctrl_T_line.set_data(t_all[s], us[s, 0])
        ctrl_tau_line.set_data(t_all[s], us[s, 1])
        lyap_line.set_data(t_all[s], Vs[s])

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), init_func=init,
        interval=max(int(dt * skip * 1000), 30), blit=False)

    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved animation -> {save_path}")


# ---------------------------------------------------------------------------
# Comparison across runs
# ---------------------------------------------------------------------------

def compare_runs(run_dirs, a_values, observer_k, observer_gamma,
                 horizon, dt, n_ics, seed, save_dir):
    """Compare multiple runs across wind values."""
    key = jax.random.PRNGKey(seed)

    # Load all models
    models = {}
    for rd in run_dirs:
        name = os.path.basename(rd)
        with open(os.path.join(rd, "policy_params.pkl"), "rb") as f:
            data = pickle.load(f)
        models[name] = data

    # Use first model's system for ICs
    first_data = list(models.values())[0]
    system = first_data.get("system", "pvtol")
    spec = get_system(system)
    x0s, _ = spec["sample_ics"](key, n_ics, 1.0, 0.0)

    # Results table: {run_name: {a_true: mean_terminal_norm}}
    table = {name: {} for name in models}

    # Build batched rollout fn per model (once), then sweep a_values
    batched_fns = {}
    for name, data in models.items():
        use_adapt = data.get("use_adapt", False)
        if use_adapt:
            adapt_cfg = AdaptiveConfig(
                adapt_enabled=True,
                use_observer=True,
                observer_k=observer_k,
                observer_gamma=observer_gamma,
                stopgrad_obs=True,
            )
        else:
            adapt_cfg = None

        batched_fns[name] = make_batched_metrics_fn(
            policy_params=data["nn"],
            lyap_params=data["lyap_params"],
            lyap_cfg=data["lyap_cfg"],
            lambda_clf=data.get("lambda_clf", 0.1),
            spec=spec,
            hidden_sizes=data.get("hidden_sizes", (64, 64)),
            adapt_cfg=adapt_cfg,
            horizon=horizon, dt=dt,
            use_shield=False,
            policy_obs_dim=data.get("obs_dim", spec["obs_dim"]),
        )

    for a_val in a_values:
        for name in models:
            norms = batched_fns[name](np.array(x0s), a_val)
            table[name][a_val] = float(np.mean(norms))

    # Print table
    print(f"\n{'=' * 70}")
    print(f"PVTOL Comparison | horizon={horizon}, dt={dt}, {n_ics} ICs")
    print(f"{'=' * 70}")
    header = f"{'Run':>35s} |"
    for a_val in a_values:
        header += f" a={a_val:4.1f} |"
    print(header)
    print("-" * len(header))
    for name in models:
        row = f"{name:>35s} |"
        for a_val in a_values:
            row += f" {table[name][a_val]:5.3f} |"
        print(row)
    print()

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(a_values))
    width = 0.8 / len(models)
    for i, name in enumerate(models):
        vals = [table[name][a] for a in a_values]
        ax.bar(x_pos + i * width, vals, width, label=name, alpha=0.8)
    ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"a={a:.1f}" for a in a_values])
    ax.set_ylabel("|xT| (terminal norm)")
    ax.set_title("PVTOL: Terminal Norm Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    chart_path = os.path.join(save_dir, "pvtol_comparison.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart -> {chart_path}")

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PVTOL validation and comparison")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Single run directory for detailed eval")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple runs")
    parser.add_argument("--run-dirs", type=str, nargs="+", default=[],
                        help="Run directories for comparison")
    parser.add_argument("--a-true", type=float, default=1.0,
                        help="Wind value for single-run eval")
    parser.add_argument("--a-values", type=float, nargs="+",
                        default=[0.0, 1.0, 2.0, 3.0],
                        help="Wind values for comparison")
    parser.add_argument("--observer-k", type=float, default=3.0)
    parser.add_argument("--observer-gamma", type=float, default=20.0)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-ics", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shield", action="store_true")
    parser.add_argument("--enforce-input-bounds", action="store_true",
                        help="Clip shield output to [u_min, u_max] after projection")
    parser.add_argument("--initial-radius", type=float, default=None,
                        help="Override initial uncertainty radius (default: from system params)")
    parser.add_argument("--animate", action="store_true",
                        help="Generate GIF animations (off by default)")
    parser.add_argument("--anim-skip", type=int, default=2)
    parser.add_argument("--anim-fps", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    if args.compare:
        save_dir = args.save_dir or "runs/pvtol_comparison"
        os.makedirs(save_dir, exist_ok=True)
        compare_runs(
            run_dirs=args.run_dirs,
            a_values=args.a_values,
            observer_k=args.observer_k,
            observer_gamma=args.observer_gamma,
            horizon=args.horizon,
            dt=args.dt,
            n_ics=args.n_ics,
            seed=args.seed,
            save_dir=save_dir,
        )
        return

    # Single-run detailed eval
    if args.run_dir is None:
        parser.error("--run-dir is required for single-run eval")

    with open(os.path.join(args.run_dir, "policy_params.pkl"), "rb") as f:
        data = pickle.load(f)

    system = data.get("system", "pvtol")
    spec = get_system(system)
    policy_params = data["nn"]
    lyap_params = data["lyap_params"]
    lyap_cfg = data["lyap_cfg"]
    lambda_clf = data.get("lambda_clf", 0.1)
    hidden_sizes = data.get("hidden_sizes", (64, 64))
    policy_obs_dim = data.get("obs_dim", spec["obs_dim"])
    use_adapt = data.get("use_adapt", False)

    if use_adapt:
        adapt_cfg = AdaptiveConfig(
            adapt_enabled=True,
            use_observer=True,
            observer_k=args.observer_k,
            observer_gamma=args.observer_gamma,
            stopgrad_obs=True,
        )
    else:
        adapt_cfg = None

    save_dir = args.save_dir or os.path.join(args.run_dir, "eval_pvtol")
    os.makedirs(save_dir, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    x0s, _ = spec["sample_ics"](key, args.n_ics, 1.0, 0.0)

    run_name = os.path.basename(args.run_dir)
    adapt_str = f"observer k={args.observer_k} gamma={args.observer_gamma}" if use_adapt else "no adaptation"
    print(f"PVTOL eval: {run_name}")
    print(f"  a_wind={args.a_true}, {adapt_str}")
    print(f"  horizon={args.horizon}, dt={args.dt}, {args.n_ics} ICs")
    print(f"  shield={'OFF' if args.no_shield else 'ON'}"
          f"{' (input bounds enforced)' if args.enforce_input_bounds else ''}")
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
        horizon=args.horizon, dt=args.dt,
        use_shield=not args.no_shield,
        policy_obs_dim=policy_obs_dim,
        enforce_input_bounds=args.enforce_input_bounds,
        initial_radius=args.initial_radius,
    )

    all_results = []
    for i, x0 in enumerate(x0s):
        res = eval_fn(np.array(x0), args.a_true)
        all_results.append(res)

        final_err = abs(res["a_hats"][-1] - args.a_true)
        final_r = res["radii"][-1]
        final_xnorm = np.linalg.norm(res["xs"][-1])
        feas_rate = np.mean(res["feasibles"])
        print(f"  IC {i}: |xT|={final_xnorm:.3f}, "
              f"|a_hat-a|={final_err:.4f}, r={final_r:.4f}, feas={feas_rate:.2f}")

        plot_pvtol_diagnostics(
            res,
            save_path=os.path.join(save_dir, f"diag_ic{i}.png"),
            title_extra=f"| {run_name}",
        )

        if args.animate:
            ic_str = f"({x0[0]:.1f},{x0[1]:.1f})"
            title = f"{run_name} | wind={args.a_true:.1f} m/s^2 | IC={ic_str}"
            animate_pvtol(
                res, dt=args.dt,
                save_path=os.path.join(save_dir, f"pvtol_ic{i}.gif"),
                title=title,
                fps=args.anim_fps,
                skip=args.anim_skip,
            )

    # Summary across ICs
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, res in enumerate(all_results):
        t = np.arange(len(res["a_hats"]))
        axes[0].plot(t, res["a_hats"], alpha=0.7, label=f"IC{i}")
        xnorms = [np.linalg.norm(res["xs"][k]) for k in range(len(res["xs"]))]
        axes[1].plot(xnorms, alpha=0.7)
        axes[2].plot(t, res["Vs"], alpha=0.7)

    axes[0].axhline(args.a_true, color="r", ls="--", lw=2)
    axes[0].set(xlabel="step", ylabel="a_hat", title="Wind estimate")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set(xlabel="step", ylabel="|x|", title="State norm")
    axes[1].grid(True, alpha=0.3)

    axes[2].set(xlabel="step", ylabel="V", title="Lyapunov V(x)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"PVTOL Summary | {run_name} | wind={args.a_true}")
    fig.tight_layout()
    summary_path = os.path.join(save_dir, "pvtol_summary.png")
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
