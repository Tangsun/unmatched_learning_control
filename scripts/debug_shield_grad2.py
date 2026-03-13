"""Debug: find at which timestep NaN gradients appear in PVTOL shield-diff rollout."""
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

# Load P3 params
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

key = jax.random.PRNGKey(42)
x0 = spec['sample_ics'](key, 1, 0.5, 0.0)[0][0]
a_true = jnp.array(1.0)  # wind

adapt_st = AdaptiveState(
    a_hat=jnp.array(0.0), info=jnp.array(1e-6),
    radius=jnp.array(0.5),
    x_hat=jnp.zeros(6), w=jnp.zeros(6), eta=jnp.zeros(6))

dt = 0.02

# Test: gradient of cumulative loss after N steps
def rollout_loss(phi, pp, n_steps):
    lp = reconstruct(phi)
    x = x0
    total = jnp.array(0.0)
    for t in range(n_steps):
        obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
        u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
        u, aux = clf_shield(u_nom, x, adapt_st, lp, lyap_cfg, clf_cfg, p,
                            affine_terms_fn=spec['affine_terms_fn'], alpha_max=0.0)
        total = total + jnp.sum(u**2)
        x = rk4_step_generic(spec['dynamics_fn'], x, u, a_true, dt, p)
        x = spec['wrap_state'](x)
    return total

print(f"Testing NaN at each horizon length (x0={x0}, a=1.0):")
print(f"{'Steps':>6s}  {'NaN?':>5s}  {'grad_norm':>10s}")
print("-" * 30)

for n in [1, 2, 5, 10, 20, 50, 100, 150, 200]:
    def loss_n(phi, pp, n=n):
        return rollout_loss(phi, pp, n)
    gi = jax.grad(loss_n, argnums=(0, 1))(lyap_phi, policy_params)
    has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi))
    gnorm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(gi)))
    print(f"{n:6d}  {'YES' if has_nan else 'no':>5s}  {float(gnorm):10.4f}")

# Also test: what happens with alpha_max=10?
print(f"\nSame test with alpha_max=10.0:")
clf_cfg2 = CLFConfig(enabled=True, lambda_clf=0.1, eps_proj=0.1)
def rollout_loss_capped(phi, pp, n_steps):
    lp = reconstruct(phi)
    x = x0
    total = jnp.array(0.0)
    for t in range(n_steps):
        obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
        u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
        u, aux = clf_shield(u_nom, x, adapt_st, lp, lyap_cfg, clf_cfg2, p,
                            affine_terms_fn=spec['affine_terms_fn'], alpha_max=10.0)
        total = total + jnp.sum(u**2)
        x = rk4_step_generic(spec['dynamics_fn'], x, u, a_true, dt, p)
        x = spec['wrap_state'](x)
    return total

for n in [1, 2, 5, 10, 20, 50, 100, 150, 200]:
    def loss_n2(phi, pp, n=n):
        return rollout_loss_capped(phi, pp, n)
    gi = jax.grad(loss_n2, argnums=(0, 1))(lyap_phi, policy_params)
    has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi))
    gnorm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(gi)))
    print(f"{n:6d}  {'YES' if has_nan else 'no':>5s}  {float(gnorm):10.4f}")

# Test: stop_gradient on the entire shield (only policy learns through nominal u)
print(f"\nTest with stop_gradient on shield correction (u = u_nom + sg(u_shield - u_nom)):")
def rollout_loss_sg_correction(phi, pp, n_steps):
    lp = reconstruct(phi)
    x = x0
    total = jnp.array(0.0)
    for t in range(n_steps):
        obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
        u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
        u_shield, aux = clf_shield(u_nom, x, adapt_st, lp, lyap_cfg, clf_cfg, p,
                            affine_terms_fn=spec['affine_terms_fn'], alpha_max=0.0)
        # Only backprop through u_nom, stop_gradient on the correction
        correction = u_shield - u_nom
        u = u_nom + jax.lax.stop_gradient(correction)
        total = total + jnp.sum(u**2)
        # But apply full u_shield for dynamics
        x = rk4_step_generic(spec['dynamics_fn'], x, u_shield, a_true, dt, p)
        x = spec['wrap_state'](x)
    return total

for n in [1, 5, 50, 200]:
    def loss_n3(phi, pp, n=n):
        return rollout_loss_sg_correction(phi, pp, n)
    gi = jax.grad(loss_n3, argnums=(0, 1))(lyap_phi, policy_params)
    has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi))
    gnorm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(gi)))
    print(f"{n:6d}  {'YES' if has_nan else 'no':>5s}  {float(gnorm):10.4f}")
