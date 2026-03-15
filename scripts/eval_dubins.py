"""Evaluate trained Dubins car policies across ICs and uncertainty levels.

Usage:
    python scripts/eval_dubins.py --run-dir runs/dubins_lyap_d4_rand
    python scripts/eval_dubins.py --run-dir runs/dubins_lyap_d4_rand --no-shield
    python scripts/eval_dubins.py --run-dir runs/dubins_lyap_d4_rand --a-values 0.0 0.3 -0.5
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


def rollout_dubins(policy_params, lyap_params, lyap_cfg, x0, spec, data,
                   a_true, use_shield, horizon=400, dt=0.05):
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
    xs, us, Vs, Vdots, feasibles, a_hats = [x0], [], [], [], [], []

    x = x0
    for _ in range(horizon):
        x = wrap_fn(x)
        obs = make_obs(x)

        if use_adapt and adapt_cfg.adapt_enabled:
            obs = jnp.concatenate([obs, jnp.array([adapt_state.a_hat, adapt_state.radius])])

        u_nom = policy_apply(policy_params, obs, u_lo, u_hi, hidden_sizes, out_dim=out_dim)

        if use_shield:
            u, aux = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=adapt_state,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_terms_fn=affine_fn,
                input_bounds=(u_lo, u_hi),
            )
            u = jnp.clip(u, u_lo, u_hi)
            feasible = float(aux["feasible"])
        else:
            u = jnp.clip(u_nom, u_lo, u_hi)
            feasible = 1.0

        V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
        xdot = dynamics_fn(x, u, a_true, p)
        Vdot = float(gradV @ xdot)

        us.append(np.array(u))
        Vs.append(float(V))
        Vdots.append(Vdot)
        feasibles.append(feasible)
        a_hats.append(float(adapt_state.a_hat))

        x = rk4_step_generic(dynamics_fn, x, u, a_true, dt, p)

        if use_adapt and adapt_cfg.adapt_enabled:
            adapt_state = adaptive_update_generic(
                adapt_state, x, u, a_true, dt, p, adapt_cfg,
                affine_terms_fn=affine_fn, dynamics_fn=dynamics_fn)

        xs.append(x)

    return {
        "xs": jnp.array(xs),
        "us": jnp.array(us),
        "Vs": jnp.array(Vs),
        "Vdots": jnp.array(Vdots),
        "feasibles": jnp.array(feasibles),
        "a_hats": jnp.array(a_hats),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--shield", action="store_true")
    parser.add_argument("--no-shield", dest="shield", action="store_false")
    parser.set_defaults(shield=True)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--a-values", type=float, nargs="+", default=[0.0, 0.3, -0.5])
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "policy_params.pkl"), "rb") as f:
        data = pickle.load(f)

    spec = get_system(data["system"])
    policy_params = data["nn"]
    lyap_params = data["lyap_params"]
    lyap_cfg = data["lyap_cfg"]

    shield_str = "shield ON" if args.shield else "no shield"
    use_adapt = data.get("use_adapt", False)
    adapt_str = " +adapt" if use_adapt else ""
    print(f"Evaluating {args.run_dir} ({shield_str}{adapt_str})")

    # ICs: 2 in-region, 2 out-of-region
    ics = {
        "IN: e=(0.5, 0.8, 0.5)": jnp.array([0.5, 0.8, 0.5]),
        "IN: e=(-1, 1, -1)": jnp.array([-1.0, 1.0, -1.0]),
        "OUT: e=(3, 3, 2.5)": jnp.array([3.0, 3.0, 2.5]),
        "OUT: e=(-4, 2, pi)": jnp.array([-4.0, 2.0, float(jnp.pi)]),
    }

    # --- Per-uncertainty-level evaluation ---
    for a_val in args.a_values:
        print(f"\n{'='*70}")
        print(f"  a_true = {a_val:.2f}")
        print(f"{'='*70}")

        results = {}
        for name, x0 in ics.items():
            res = rollout_dubins(
                policy_params, lyap_params, lyap_cfg, x0, spec, data,
                a_true=a_val, use_shield=args.shield,
                horizon=args.horizon, dt=args.dt)
            results[name] = res

        # Print table
        print(f"\n{'IC':<28s} | {'|e_f|':>8s} | {'V_f':>10s} | "
              f"{'V decr%':>8s} | {'feas%':>7s}")
        print("-" * 75)
        for name, res in results.items():
            xT = res["xs"][-1]
            norm_f = float(jnp.linalg.norm(xT))
            V_f = float(res["Vs"][-1])
            v_decr = float(jnp.mean(jnp.array(res["Vdots"]) < 0)) * 100
            feas = float(jnp.mean(res["feasibles"])) * 100
            print(f"{name:<28s} | {norm_f:8.4f} | {V_f:10.4f} | "
                  f"{v_decr:7.1f}% | {feas:6.1f}%")

        # Plot
        t = jnp.arange(args.horizon) * args.dt
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f"Dubins Eval | a={a_val:.2f} | {shield_str}{adapt_str}\n"
                     f"{os.path.basename(args.run_dir)}", fontsize=13)
        colors = plt.cm.tab10.colors

        for i, (name, res) in enumerate(results.items()):
            c = colors[i % len(colors)]
            style = "--" if "OUT" in name else "-"

            # e_x, e_y
            axes[0, 0].plot(t, res["xs"][:-1, 0], style, color=c, label=name, alpha=0.8)
            axes[0, 1].plot(t, res["xs"][:-1, 1], style, color=c, alpha=0.8)
            # v, omega
            axes[1, 0].plot(t, res["us"][:, 0], style, color=c, alpha=0.8)
            axes[1, 1].plot(t, res["us"][:, 1], style, color=c, alpha=0.8)
            # V, Vdot + lambda*V
            axes[2, 0].plot(t, res["Vs"], style, color=c, alpha=0.8)
            constraint = jnp.array(res["Vdots"]) + data["lambda_clf"] * jnp.array(res["Vs"])
            axes[2, 1].plot(t, constraint, style, color=c, alpha=0.8)

        axes[0, 0].set(ylabel="e_x", title="Along-track error")
        axes[0, 0].axhline(0, color="k", lw=0.5)
        axes[0, 0].legend(fontsize=7, loc="upper right")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set(ylabel="e_y", title="Lateral error")
        axes[0, 1].axhline(0, color="k", lw=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set(ylabel="v", title="Forward speed")
        axes[1, 0].axhline(spec["params"].v_ref, color="gray", ls=":", lw=0.8, label="v_ref")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=7)

        axes[1, 1].set(ylabel="omega", title="Turn rate")
        axes[1, 1].axhline(0, color="k", lw=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        axes[2, 0].set(xlabel="time (s)", ylabel="V", title="Lyapunov V")
        axes[2, 0].set_yscale("symlog", linthresh=0.01)
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].set(xlabel="time (s)", ylabel="dV/dt + lambda*V",
                       title="CLF constraint (should be <= 0)")
        axes[2, 1].axhline(0, color="r", lw=0.5, ls="--")
        axes[2, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        a_tag = f"a{a_val:.2f}".replace(".", "p").replace("-", "m")
        suffix = "shield" if args.shield else "noshield"
        out_path = os.path.join(args.run_dir, f"eval_dubins_{a_tag}_{suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved -> {out_path}")

    # If adaptive, plot a_hat convergence for all ICs at a=0.3
    if use_adapt:
        a_val = 0.3
        t = jnp.arange(args.horizon) * args.dt
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"Adaptive estimator convergence (a_true={a_val})")
        colors = plt.cm.tab10.colors
        for i, (name, x0) in enumerate(ics.items()):
            res = rollout_dubins(
                policy_params, lyap_params, lyap_cfg, x0, spec, data,
                a_true=a_val, use_shield=args.shield,
                horizon=args.horizon, dt=args.dt)
            style = "--" if "OUT" in name else "-"
            ax.plot(t, res["a_hats"], style, color=colors[i], label=name, alpha=0.8)
        ax.axhline(a_val, color="r", ls="--", lw=1.5, label=f"a_true={a_val}")
        ax.set(xlabel="time (s)", ylabel="a_hat")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(args.run_dir, "eval_dubins_adapt_convergence.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
