"""Quick eval: compare E2 vs E3 (dubins) and P3 vs P4b (pvtol) with shield ON/OFF."""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import jax
import jax.numpy as jnp
from adaptive_clf.configs import AdaptiveConfig, AdaptiveState, CLFConfig, LyapunovConfig
from adaptive_clf.adaptive import adaptive_update_observer, init_adaptive_state
from adaptive_clf.systems import get_system
from adaptive_clf.lyapunov import lyapunov_value_and_grad
from adaptive_clf.shield import clf_shield
from adaptive_clf.nn import policy_apply
from adaptive_clf.integrator import rk4_step_generic


def eval_run(run_dir, a_values, use_shield=True, n_ics=6, horizon=400, dt=0.05):
    with open(os.path.join(run_dir, 'policy_params.pkl'), 'rb') as f:
        d = pickle.load(f)
    spec = get_system(d['system'])
    p = spec['params']
    lyap_cfg = d['lyap_cfg']
    lyap_params = d['lyap_params']
    hidden = d['hidden_sizes']
    policy_params = d['nn']
    use_adapt = d.get('use_adapt', False)
    adapt_cfg = d.get('adapt_cfg', None)
    lambda_clf = d.get('lambda_clf', 0.1)
    ctrl_dim = d.get('ctrl_dim', 2)

    clf_cfg = CLFConfig(enabled=use_shield, lambda_clf=lambda_clf, eps_proj=0.1)

    results = {}
    for a_val in a_values:
        key = jax.random.PRNGKey(42)
        ics = spec['sample_ics'](key, n_ics, 0.5, 0.0)[0]

        term_norms = []
        for i in range(n_ics):
            x = ics[i]
            state_dim = spec['state_dim']
            a_true = jnp.array(a_val)

            if use_adapt and adapt_cfg is not None:
                adapt_st = init_adaptive_state(p, adapt_cfg, state_dim=state_dim, x0=x)
            else:
                adapt_st = AdaptiveState(
                    a_hat=jnp.array(0.0), info=jnp.array(1e-6),
                    radius=jnp.array(0.0),
                    x_hat=jnp.zeros(state_dim),
                    w=jnp.zeros(state_dim),
                    eta=jnp.zeros(state_dim))

            for t in range(horizon):
                obs = spec['make_obs'](x)
                if use_adapt:
                    obs = jnp.concatenate([obs, jnp.array([adapt_st.a_hat, adapt_st.radius])])

                u_nom = policy_apply(policy_params, obs, spec['u_min'], spec['u_max'],
                                     hidden, out_dim=ctrl_dim)

                if use_shield:
                    u, _ = clf_shield(u_nom, x, adapt_st, lyap_params, lyap_cfg, clf_cfg, p,
                                      affine_terms_fn=spec['affine_terms_fn'])
                else:
                    u = jnp.clip(u_nom, spec['u_min'], spec['u_max'])

                x = rk4_step_generic(spec['dynamics_fn'], x, u, a_true, dt, p)
                x = spec['wrap_state'](x)

                if use_adapt and adapt_cfg is not None:
                    adapt_st = adaptive_update_observer(adapt_st, x, u, dt, p, adapt_cfg,
                                                        affine_terms_fn=spec['affine_terms_fn'])

            term_norms.append(float(jnp.linalg.norm(x)))

        results[a_val] = sum(term_norms) / len(term_norms)
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
        # Add P4b if it exists
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
