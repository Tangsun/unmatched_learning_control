#!/usr/bin/env python
"""Run swing-up experiments with gradual curriculum for acrobot.

Usage (from repo root):
    python scripts/run_swingup_experiments.py --quick
    python scripts/run_swingup_experiments.py --medium
    python scripts/run_swingup_experiments.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np

from adaptive_clf.configs import AcrobotParams
from adaptive_clf.acrobot import solve_lqr_P, acrobot_dynamics_true, rk4_step
from adaptive_clf.train import train_swingup


def evaluate_policy(load_dir, n_eval=8, duration=10.0, use_lqr_blend=True):
    """Quick evaluation: return dict of summary stats.

    If use_lqr_blend is True, blends NN output with LQR near upright,
    implementing approach (1): LQR handles final stabilization.
    """
    from adaptive_clf.nn import policy_apply
    from adaptive_clf.adaptive import init_adaptive_state, make_policy_observation
    from adaptive_clf.configs import AdaptiveConfig

    p = AcrobotParams()
    dt = 0.005
    n_steps = int(duration / dt)

    with open(os.path.join(load_dir, "policy_params.pkl"), "rb") as f:
        policy_params = jax.tree.map(jnp.asarray, pickle.load(f))

    adapt_cfg = AdaptiveConfig()
    hidden_sizes = (64, 64)

    Q_lqr = jnp.diag(jnp.array([40.0, 40.0, 8.0, 8.0]))
    R_lqr = jnp.array([[0.5]])
    P_lqr, K_lqr = solve_lqr_P(p, Q=Q_lqr, R=R_lqr, a_nom=0.0)
    lqr_threshold = 0.08

    ics = [
        ("hang-down",     jnp.array([-jnp.pi, 0.0, 0.0, 0.0])),
        ("hang+vel",      jnp.array([-jnp.pi, 0.0, 1.0, 0.0])),
        ("horizontal",    jnp.array([-jnp.pi / 2, 0.0, 0.0, 0.0])),
        ("3/4 down",      jnp.array([-3 * jnp.pi / 4, 0.0, 0.0, 0.0])),
        ("near 0.3",      jnp.array([0.3, 0.0, 0.0, 0.0])),
        ("near 0.1",      jnp.array([0.1, 0.0, 0.0, 0.0])),
        ("near 0.05",     jnp.array([0.05, 0.0, 0.0, 0.0])),
        ("near 0.01",     jnp.array([0.01, 0.0, 0.0, 0.0])),
    ][:n_eval]

    results = {}
    results_no_lqr = {}
    for label, x0 in ics:
        for use_lqr in ([True, False] if use_lqr_blend else [False]):
            adaptive_state = init_adaptive_state(p, adapt_cfg)
            x = x0
            a_true_arr = jnp.array(0.0)
            for _ in range(n_steps):
                state_norm = float(jnp.linalg.norm(x))
                if use_lqr and state_norm < lqr_threshold:
                    u = jnp.clip(-(K_lqr @ x).squeeze(), p.u_min, p.u_max)
                else:
                    obs = make_policy_observation(x, adaptive_state, adapt_cfg)
                    u = policy_apply(policy_params, obs, p.u_min, p.u_max, hidden_sizes)
                    u = jnp.clip(u, p.u_min, p.u_max)
                x = rk4_step(acrobot_dynamics_true, x, u, a_true_arr, dt, p)
            fn = float(jnp.linalg.norm(x))
            if use_lqr:
                results[label] = fn
            else:
                results_no_lqr[label] = fn

    if not results:
        results = results_no_lqr

    stabilize_ok = sum(1 for k, v in results.items() if "near" in k and v < 0.5)
    swing_ok = sum(1 for k, v in results.items() if "near" not in k and v < 1.0)

    out = {
        "all": results,
        "mean_near": np.mean([v for k, v in results.items() if "near" in k]),
        "mean_far": np.mean([v for k, v in results.items() if "near" not in k]),
        "stabilize_success": stabilize_ok,
        "swing_success": swing_ok,
    }
    if results_no_lqr:
        out["all_no_lqr"] = results_no_lqr
    return out


def make_phases(n_total, base_lr, n_substeps, clf, terminal_cost=50.0,
                Q_scale=0.1, sampling="mixed", bptt_window=0, R_energy=0.0):
    """Create gradual curriculum phases with short→long horizon and mixed sampling.

    Key idea: start with very short horizons (5 steps = 0.1s) to get
    meaningful gradients, then gradually increase both horizon and region.
    Energy shaping is applied in later phases to guide swing-up learning.
    """
    schedule = [
        # (region_scale, horizon, bptt_window, R_energy_mult)
        (0.003,   5,   0,  0.0),
        (0.005,  10,   0,  0.0),
        (0.01,   15,   0,  0.0),
        (0.02,   25,  10,  0.0),
        (0.05,   40,  15,  0.5),
        (0.1,    60,  20,  1.0),
        (0.2,   100,  30,  1.0),
        (0.5,   150,  40,  1.0),
    ]
    n_phases = min(len(schedule), max(3, n_total // 40))
    schedule = schedule[:n_phases]

    steps_each = n_total // n_phases
    last_steps = n_total - steps_each * (n_phases - 1)

    phases = []
    for i, (rs, h, bw, e_mult) in enumerate(schedule):
        n_s = last_steps if i == n_phases - 1 else steps_each
        lr_decay = max(0.3, 1.0 - 0.08 * i)
        phases.append({
            "n_steps": n_s,
            "sampling": sampling,
            "region_scale": rs,
            "horizon": h,
            "bptt_window": bw if bptt_window == 0 else bptt_window,
            "terminal_cost_scale": terminal_cost,
            "Q_stage_scale": Q_scale,
            "R_energy": R_energy * e_mult,
            "lr": base_lr * lr_decay,
            "u_clip": 100.0,
        })
    return phases


EXPERIMENTS = {
    "A_energy_mixed": {
        "desc": "Energy shaping + mixed sampling + short→long horizon",
        "clf": False,
        "sampling": "mixed",
        "R_energy": 0.5,
    },
    "B_energy_curriculum": {
        "desc": "Energy shaping + curriculum + short→long horizon",
        "clf": False,
        "sampling": "curriculum",
        "R_energy": 0.5,
    },
    "C_no_energy_mixed": {
        "desc": "No energy shaping + mixed sampling (baseline)",
        "clf": False,
        "sampling": "mixed",
        "R_energy": 0.0,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (300 steps)")
    parser.add_argument("--medium", action="store_true",
                        help="Medium run (1000 steps)")
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.quick:
        n_total, batch_size, horizon, base_lr = 400, 32, 60, 3e-3
    elif args.medium:
        n_total, batch_size, horizon, base_lr = 1200, 64, 100, 1e-3
    else:
        n_total, batch_size, horizon, base_lr = 4000, 64, 150, 5e-4

    n_substeps = 4

    exps = args.experiments or list(EXPERIMENTS.keys())
    results = {}

    for name in exps:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}, skipping")
            continue
        exp = EXPERIMENTS[name]
        save_dir = f"runs/exp_{name}"
        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT: {name}")
        print(f"  {exp['desc']}")
        print(f"{'=' * 70}")

        u_clip_val = exp.get("u_clip", 100.0)
        samp = exp.get("sampling", "mixed")
        r_energy = exp.get("R_energy", 0.0)
        phases = make_phases(
            n_total, base_lr, n_substeps,
            clf=exp["clf"],
            terminal_cost=50.0,
            Q_scale=0.05,
            sampling=samp,
            R_energy=r_energy,
        )

        t0 = time.time()
        try:
            train_swingup(
                n_steps=n_total,
                lr=base_lr,
                batch_size=batch_size,
                horizon=horizon,
                max_grad_norm=1.0,
                log_every=max(n_total // 20, 10),
                save_dir=save_dir,
                seed=args.seed,
                fixed_batch=False,
                clf_enabled=exp["clf"],
                lqr_blend=False,
                terminal_cost_scale=50.0,
                Q_stage_scale=0.05,
                u_clip=u_clip_val,
                n_substeps=n_substeps,
                phases=phases,
            )
            elapsed = time.time() - t0
            eval_res = evaluate_policy(save_dir, use_lqr_blend=True)
            eval_res["train_time"] = elapsed
            results[name] = eval_res
            print(f"\n  >> {name}: time={elapsed:.0f}s")
            print(f"     {'IC':15s}  {'NN+LQR':>10s}  {'NN only':>10s}")
            for k, v in eval_res["all"].items():
                v2 = eval_res.get("all_no_lqr", {}).get(k, float('nan'))
                ok = "OK" if v < 0.5 else ""
                print(f"     {k:15s}: {v:10.3f}   {v2:10.3f} {ok}")
        except Exception as e:
            print(f"\n  >> {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    print(f"\n\n{'=' * 70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    fmt = "{:<25s} {:>10s} {:>10s} {:>10s} {:>8s}"
    print(fmt.format("Name", "Near mean", "Far mean", "Stab OK", "Time"))
    print("-" * 70)
    for name in exps:
        if name not in results:
            continue
        r = results[name]
        if "error" in r:
            print(f"{name:<25s} {'FAILED':>10s}")
        else:
            print(f"{name:<25s} {r['mean_near']:10.3f} {r['mean_far']:10.3f} "
                  f"{r['stabilize_success']:>5d}/4   "
                  f"{r.get('train_time', 0):7.0f}s")

    summary_path = "runs/experiment_summary.pkl"
    os.makedirs("runs", exist_ok=True)
    with open(summary_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
