"""Debug: scan many ICs and a_true values to find NaN in PVTOL shield-diff."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import jax
import jax.numpy as jnp
import pickle

from adaptive_clf.configs import AdaptiveState, CLFConfig
from adaptive_clf.systems import get_system
from adaptive_clf.lyapunov import lyapunov_value_and_grad
from adaptive_clf.shield import clf_shield
from adaptive_clf.nn import policy_apply
from adaptive_clf.integrator import rk4_step_generic

with open('runs/pvtol_p3_observer_scratch/policy_params.pkl', 'rb') as f:
    d = pickle.load(f)

spec = get_system('pvtol')
p = spec['params']
lyap_cfg = d['lyap_cfg']
lyap_params = d['lyap_params']
hidden = d['hidden_sizes']
policy_params = d['nn']
clf_cfg = CLFConfig(enabled=True, lambda_clf=0.1, eps_proj=0.1)

lyap_static = {k: v for k, v in lyap_params.items() if k in ("mode", "x_eq")}
lyap_phi = lyap_params["phi"]

def reconstruct(phi):
    return {**lyap_static, "phi": phi}

adapt_st = AdaptiveState(
    a_hat=jnp.array(0.0), info=jnp.array(1e-6),
    radius=jnp.array(0.5),
    x_hat=jnp.zeros(6), w=jnp.zeros(6), eta=jnp.zeros(6))

dt = 0.02
horizon = 200

def rollout_loss(phi, pp, x0, a_true):
    lp = reconstruct(phi)
    x = x0
    total = jnp.array(0.0)
    for t in range(horizon):
        obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
        u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
        u, _ = clf_shield(u_nom, x, adapt_st, lp, lyap_cfg, clf_cfg, p,
                          affine_terms_fn=spec['affine_terms_fn'], alpha_max=0.0)
        total = total + jnp.sum(u**2)
        x = rk4_step_generic(spec['dynamics_fn'], x, u, a_true, dt, p)
        x = spec['wrap_state'](x)
    return total

# Test batched like training: vmap over ICs
# The training uses jax.lax.scan not python for-loop, but let's test with the for-loop first
# to match the actual training behavior

print("Scanning ICs × a_true for NaN (for-loop rollout, 200 steps):")
print(f"{'IC':>4s}  {'a_true':>7s}  {'NaN?':>5s}  {'grad_norm':>10s}")
print("-" * 40)

# Use same seed as training for ICs, test multiple a_true values
nan_found = 0
for seed in range(3):
    key = jax.random.PRNGKey(seed)
    ics = spec['sample_ics'](key, 8, 0.5, 0.0)[0]
    a_values = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5]

    for i in range(8):
        for a_val in a_values:
            x0 = ics[i]
            a_t = jnp.array(a_val)

            def loss_fn(phi, pp, x0=x0, a_t=a_t):
                return rollout_loss(phi, pp, x0, a_t)

            gi = jax.grad(loss_fn, argnums=(0, 1))(lyap_phi, policy_params)
            has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi))
            gnorm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(gi)))

            if has_nan:
                nan_found += 1
                print(f"s{seed}i{i:2d}  {a_val:7.1f}  {'YES':>5s}  {float(gnorm):10.4f}  x0={x0}")
            # Only print clean ones occasionally
            elif i == 0 and a_val == 0.0:
                print(f"s{seed}i{i:2d}  {a_val:7.1f}  {'no':>5s}  {float(gnorm):10.4f}")

    print(f"  [seed {seed}: checked {8*len(a_values)} combos, {nan_found} NaN so far]")

print(f"\nTotal NaN: {nan_found}/{3*8*6}")

if nan_found == 0:
    print("\nNo NaN found with for-loop. Testing with jax.lax.scan (like training)...")
    # The training uses jax.lax.scan, which may behave differently w.r.t. gradient accumulation
    def rollout_scan(phi, pp, x0, a_true):
        lp = reconstruct(phi)
        def body(carry, _):
            x = carry
            obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
            u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
            u, _ = clf_shield(u_nom, x, adapt_st, lp, lyap_cfg, clf_cfg, p,
                              affine_terms_fn=spec['affine_terms_fn'], alpha_max=0.0)
            cost = jnp.sum(u**2)
            x_next = rk4_step_generic(spec['dynamics_fn'], x, u, a_true, dt, p)
            x_next = spec['wrap_state'](x_next)
            return x_next, cost
        xT, costs = jax.lax.scan(body, x0, None, length=horizon)
        return jnp.sum(costs)

    key = jax.random.PRNGKey(0)
    ics = spec['sample_ics'](key, 4, 0.5, 0.0)[0]
    print(f"{'IC':>4s}  {'a_true':>7s}  {'NaN?':>5s}  {'grad_norm':>10s}")
    for i in range(4):
        for a_val in [0.0, 1.0, 1.5]:
            x0 = ics[i]
            a_t = jnp.array(a_val)
            def loss_scan(phi, pp, x0=x0, a_t=a_t):
                return rollout_scan(phi, pp, x0, a_t)
            gi = jax.grad(loss_scan, argnums=(0, 1))(lyap_phi, policy_params)
            has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi))
            gnorm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(gi)))
            status = "YES" if has_nan else "no"
            print(f"{i:4d}  {a_val:7.1f}  {status:>5s}  {float(gnorm):10.4f}")
