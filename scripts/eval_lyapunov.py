"""Evaluate a trained policy + neural Lyapunov with optional CLF shield."""

import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from adaptive_clf.configs import AdaptiveState, CLFConfig, LyapunovConfig
from adaptive_clf.integrator import rk4_step_generic
from adaptive_clf.lyapunov import lyapunov_value_and_grad
from adaptive_clf.nn import policy_apply
from adaptive_clf.shield import clf_shield
from adaptive_clf.systems import get_system


def rollout_eval(policy_params, lyap_params, lyap_cfg, x0, spec,
                 hidden_sizes, lambda_clf, use_shield, horizon=400, dt=0.02):
    p = spec["params"]
    wrap_fn = spec["wrap_state"]
    dynamics_fn = spec["dynamics_fn"]
    affine_fn = spec["affine_terms_fn"]
    make_obs = spec["make_obs"]
    u_min, u_max = p.u_min, p.u_max

    clf_cfg = CLFConfig(enabled=True, lambda_clf=lambda_clf)
    dummy_adaptive = AdaptiveState(
        a_hat=jnp.array(0.0), info=jnp.array(1e-6), radius=jnp.array(0.0))

    xs, us, Vs, Vdots, feasibles = [x0], [], [], [], []
    x = x0
    a_true = jnp.array(0.0)

    for _ in range(horizon):
        x = jnp.clip(x, -20.0, 20.0)
        x = wrap_fn(x)
        obs = make_obs(x)
        u_nom = policy_apply(policy_params, obs, u_min, u_max, hidden_sizes)

        if use_shield:
            u, aux = clf_shield(
                u_nom=u_nom, x=x, adaptive_state=dummy_adaptive,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_terms_fn=affine_fn,
            )
            u = jnp.clip(u, u_min, u_max)
            feasible = float(aux["feasible"])
        else:
            u = jnp.clip(u_nom, u_min, u_max)
            feasible = 1.0

        V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
        xdot = dynamics_fn(x, u, a_true, p)
        Vdot = float(gradV @ xdot)

        us.append(float(u.squeeze()))
        Vs.append(float(V))
        Vdots.append(Vdot)
        feasibles.append(feasible)

        x = rk4_step_generic(dynamics_fn, x, u, a_true, dt, p)
        xs.append(x)

    xs = jnp.array(xs)
    return {
        "xs": xs,
        "us": jnp.array(us),
        "Vs": jnp.array(Vs),
        "Vdots": jnp.array(Vdots),
        "feasibles": jnp.array(feasibles),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--shield", action="store_true")
    parser.add_argument("--no-shield", dest="shield", action="store_false")
    parser.set_defaults(shield=True)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    # Load
    params_path = os.path.join(args.run_dir, "policy_params.pkl")
    with open(params_path, "rb") as f:
        data = pickle.load(f)

    spec = get_system(data["system"])
    policy_params = data["nn"]
    lyap_params = data["lyap_params"]
    lyap_cfg = data["lyap_cfg"]
    hidden_sizes = data["hidden_sizes"]
    lambda_clf = data["lambda_clf"]

    shield_str = "shield ON" if args.shield else "no shield"
    print(f"Evaluating {args.run_dir} ({shield_str})")

    # Test ICs
    ics = {
        "small theta=0.47": jnp.array([0.0, 0.47, 0.0, 0.0]),
        "boundary theta=0.94": jnp.array([0.0, 0.94, 0.0, 0.0]),
        "with velocity": jnp.array([0.0, 0.75, 0.0, 0.3]),
        "OUT: theta=pi": jnp.array([0.0, jnp.pi, 0.0, 0.0]),
        "OUT: theta=2.0": jnp.array([0.0, 2.0, 0.0, 0.0]),
    }

    results = {}
    for name, x0 in ics.items():
        res = rollout_eval(
            policy_params, lyap_params, lyap_cfg, x0, spec,
            hidden_sizes, lambda_clf, args.shield, args.horizon, args.dt)
        results[name] = res

    # Print table
    print(f"\n{'IC':<35s} | {'final theta':>12s} | {'final |x|':>10s} | "
          f"{'final V':>10s} | {'V decr%':>10s} | {'feasible%':>10s}")
    print("-" * 100)
    for name, res in results.items():
        xT = res["xs"][-1]
        theta_f = float(xT[1])
        norm_f = float(jnp.linalg.norm(xT))
        V_f = float(res["Vs"][-1])
        v_decr = float(jnp.mean(jnp.array(res["Vdots"]) < 0)) * 100
        feas = float(jnp.mean(res["feasibles"])) * 100
        print(f"{name:<35s} | {theta_f:12.4f} | {norm_f:10.4f} | "
              f"{V_f:10.6f} | {v_decr:9.1f}% | {feas:9.1f}%")

    # Plot
    t = jnp.arange(args.horizon) * args.dt
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Neural Lyapunov Eval ({shield_str})", fontsize=14)
    colors = plt.cm.tab10.colors

    for i, (name, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        style = "--" if "OUT" in name else "-"

        axes[0, 0].plot(t, res["xs"][:-1, 1], style, color=c, label=name, alpha=0.8)
        axes[0, 1].plot(t, res["xs"][:-1, 0], style, color=c, alpha=0.8)
        axes[1, 0].plot(t, res["us"], style, color=c, alpha=0.8)
        axes[1, 1].plot(t, res["Vs"], style, color=c, alpha=0.8)
        axes[2, 0].plot(t, res["Vdots"], style, color=c, alpha=0.8)
        constraint = jnp.array(res["Vdots"]) + lambda_clf * jnp.array(res["Vs"])
        axes[2, 1].plot(t, constraint, style, color=c, alpha=0.8)

    axes[0, 0].set(ylabel="theta (rad)", title="Pole angle (0=upright)")
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set(ylabel="x_cart", title="Cart position")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set(ylabel="u", title="Control force")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set(ylabel="V", title="Lyapunov V")
    axes[1, 1].set_yscale("symlog", linthresh=0.01)
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].set(xlabel="time (s)", ylabel="dV/dt", title="Lyapunov time derivative")
    axes[2, 0].axhline(0, color="r", lw=0.5, ls="--")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].set(xlabel="time (s)", ylabel="dV/dt + λV",
                    title="CLF constraint (should be ≤ 0)")
    axes[2, 1].axhline(0, color="r", lw=0.5, ls="--", label="feasibility boundary")
    axes[2, 1].legend(fontsize=7)
    axes[2, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    suffix = "shield" if args.shield else "noshield"
    out_path = os.path.join(args.run_dir, f"eval_lyapunov_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
