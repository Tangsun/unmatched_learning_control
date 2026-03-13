"""Quick eval: compare E2 vs E3 (dubins) and P3 vs P4b (pvtol) with shield ON/OFF.

Uses compiled lax.scan + vmap rollouts for fast evaluation.
"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import jax
import numpy as np
from adaptive_clf.systems import get_system
from adaptive_clf.eval_rollout import make_batched_metrics_fn


def eval_run(run_dir, a_values, use_shield=True, n_ics=6, horizon=400, dt=0.05):
    with open(os.path.join(run_dir, 'policy_params.pkl'), 'rb') as f:
        d = pickle.load(f)
    spec = get_system(d['system'])
    lyap_cfg = d['lyap_cfg']
    lyap_params = d['lyap_params']
    hidden = d['hidden_sizes']
    policy_params = d['nn']
    use_adapt = d.get('use_adapt', False)
    adapt_cfg = d.get('adapt_cfg', None)
    lambda_clf = d.get('lambda_clf', 0.1)
    policy_obs_dim = d.get('obs_dim', spec['obs_dim'])

    key = jax.random.PRNGKey(42)
    ics = np.array(spec['sample_ics'](key, n_ics, 0.5, 0.0)[0])

    # Build batched rollout once, reuse for all a_values
    batched_fn = make_batched_metrics_fn(
        policy_params, lyap_params, lyap_cfg, lambda_clf,
        spec, hidden, adapt_cfg if use_adapt else None,
        horizon=horizon, dt=dt, use_shield=use_shield,
        policy_obs_dim=policy_obs_dim,
    )

    results = {}
    for a_val in a_values:
        norms = batched_fn(ics, a_val)
        results[a_val] = float(np.mean(norms))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="dubins", choices=["dubins", "pvtol"])
    args = parser.parse_args()

    if args.system == "dubins":
        runs = {
            'E2 (soft-only)': 'runs/dubins_scratch_e2',
            'E3 (shield-ft)': 'runs/dubins_e3_shield_finetune',
        }
        a_vals = [0.1, 0.3, 0.5, 0.7]
    else:
        runs = {
            'P3 (soft-only)': 'runs/pvtol_p3_observer_scratch',
        }
        if os.path.exists('runs/pvtol_p4b_shield_alphamax/policy_params.pkl'):
            runs['P4b (shield-ft)'] = 'runs/pvtol_p4b_shield_alphamax'
        a_vals = [0.0, 0.5, 1.0, 1.5, 2.0]

    for shield_mode in [True, False]:
        mode_str = "Shield ON" if shield_mode else "Shield OFF"
        print(f"\n{'='*60}")
        print(f"  {mode_str}")
        print(f"{'='*60}")
        header = f"{'Run':25s}  " + "  ".join(f"a={a:.1f}" for a in a_vals)
        print(header)
        print("-" * len(header))
        for name, path in runs.items():
            t0 = time.time()
            r = eval_run(path, a_vals, use_shield=shield_mode)
            elapsed = time.time() - t0
            row = f"{name:25s}  " + "  ".join(f"{r[a]:.3f}" for a in a_vals)
            print(f"{row}  ({elapsed:.1f}s)")
