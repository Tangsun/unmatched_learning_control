"""Debug: find where NaN gradients come from in PVTOL shield-diff."""
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

# Split lyap into static + differentiable
lyap_static = {k: v for k, v in lyap_params.items() if k in ("mode", "x_eq")}
lyap_phi = lyap_params["phi"]

def reconstruct(phi):
    return {**lyap_static, "phi": phi}

# Sample ICs
key = jax.random.PRNGKey(42)
ics = spec['sample_ics'](key, 20, 0.5, 0.0)[0]
adapt_st = AdaptiveState(
    a_hat=jnp.array(0.0), info=jnp.array(1e-6),
    radius=jnp.array(0.5),
    x_hat=jnp.zeros(6), w=jnp.zeros(6), eta=jnp.zeros(6))

# Test each IC for NaN
print("Testing gradient through shield projection for 20 ICs:")
print(f"{'IC':>4s}  {'NaN?':>5s}  {'V':>8s}  {'|LgV|':>8s}  {'||LgV||²':>10s}  {'violation':>10s}  {'alpha':>8s}  {'b_scalar':>10s}")
print("-" * 80)

nan_count = 0
for i in range(20):
    xi = ics[i]

    # Forward pass to get intermediate values
    obs = jnp.concatenate([spec['make_obs'](xi), jnp.array([adapt_st.a_hat, adapt_st.radius])])
    u_nom = policy_apply(policy_params, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
    u, aux = clf_shield(u_nom, xi, adapt_st, lyap_params, lyap_cfg, clf_cfg, p,
                        affine_terms_fn=spec['affine_terms_fn'], alpha_max=0.0)

    # Backward pass
    def loss_fn(phi, pp, x=xi):
        lp = reconstruct(phi)
        obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
        u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
        u, _ = clf_shield(u_nom, x, adapt_st, lp, lyap_cfg, clf_cfg, p,
                          affine_terms_fn=spec['affine_terms_fn'], alpha_max=0.0)
        return jnp.sum(u**2)

    gi = jax.grad(loss_fn, argnums=(0, 1))(lyap_phi, policy_params)
    has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi))

    V = float(aux['V'])
    LgV_norm = float(jnp.linalg.norm(aux['LgV']))
    LgV_sq = float(aux['a_norm_sq'])
    viol = float(aux['violation'])
    alpha = float(aux['projection_gain'])
    b = float(aux['b_scalar'])

    status = "YES" if has_nan else "no"
    if has_nan:
        nan_count += 1
    print(f"{i:4d}  {status:>5s}  {V:8.4f}  {LgV_norm:8.4f}  {LgV_sq:10.6f}  {viol:10.4f}  {alpha:8.4f}  {b:10.4f}")

print(f"\n{nan_count}/20 ICs produce NaN gradients")

# If any NaN, test with different eps_proj values
if nan_count > 0:
    # Pick a NaN IC
    nan_ic = None
    for i in range(20):
        xi = ics[i]
        def loss_fn2(phi, pp, x=xi):
            lp = reconstruct(phi)
            obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
            u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
            u, _ = clf_shield(u_nom, x, adapt_st, lp, lyap_cfg, clf_cfg, p,
                              affine_terms_fn=spec['affine_terms_fn'], alpha_max=0.0)
            return jnp.sum(u**2)
        gi = jax.grad(loss_fn2, argnums=(0, 1))(lyap_phi, policy_params)
        if any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi)):
            nan_ic = i
            break

    if nan_ic is not None:
        xi = ics[nan_ic]
        print(f"\nDiagnosing IC {nan_ic} with different fixes:")

        # Test: stop_gradient on alpha (only backprop through LgV direction, not magnitude)
        def loss_sg_alpha(phi, pp, x=xi):
            lp = reconstruct(phi)
            obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
            u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
            # Manual shield: compute projection with stop_gradient on alpha
            V, gradV = lyapunov_value_and_grad(lp, lyap_cfg, x)
            f_, G_, Y_ = spec['affine_terms_fn'](x, p)
            LfV = gradV @ f_
            LgV = gradV @ G_
            LyV = gradV @ Y_
            b = -0.1 * V - LfV - LyV * adapt_st.a_hat - jnp.abs(LyV) * adapt_st.radius
            violation = jnp.dot(LgV, u_nom) - b
            denom = jnp.maximum(jnp.dot(LgV, LgV), 0.1)
            alpha = jax.nn.relu(violation) / denom
            alpha = jax.lax.stop_gradient(alpha)  # <-- stop grad on alpha
            u = u_nom - alpha * LgV
            u = jnp.clip(u, spec['u_min'], spec['u_max'])
            return jnp.sum(u**2)

        gi_sg = jax.grad(loss_sg_alpha, argnums=(0, 1))(lyap_phi, policy_params)
        has_nan_sg = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi_sg))
        print(f"  stop_gradient(alpha): NaN = {has_nan_sg}")

        # Test: stop_gradient on LgV (only backprop through alpha magnitude)
        def loss_sg_lgv(phi, pp, x=xi):
            lp = reconstruct(phi)
            obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
            u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
            V, gradV = lyapunov_value_and_grad(lp, lyap_cfg, x)
            f_, G_, Y_ = spec['affine_terms_fn'](x, p)
            LfV = gradV @ f_
            LgV = gradV @ G_
            LyV = gradV @ Y_
            b = -0.1 * V - LfV - LyV * adapt_st.a_hat - jnp.abs(LyV) * adapt_st.radius
            violation = jnp.dot(LgV, u_nom) - b
            denom = jnp.maximum(jnp.dot(LgV, LgV), 0.1)
            alpha = jax.nn.relu(violation) / denom
            LgV_sg = jax.lax.stop_gradient(LgV)  # <-- stop grad on direction
            u = u_nom - alpha * LgV_sg
            u = jnp.clip(u, spec['u_min'], spec['u_max'])
            return jnp.sum(u**2)

        gi_sg2 = jax.grad(loss_sg_lgv, argnums=(0, 1))(lyap_phi, policy_params)
        has_nan_sg2 = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi_sg2))
        print(f"  stop_gradient(LgV direction): NaN = {has_nan_sg2}")

        # Test: both stopped
        def loss_sg_both(phi, pp, x=xi):
            lp = reconstruct(phi)
            obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
            u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
            V, gradV = lyapunov_value_and_grad(lp, lyap_cfg, x)
            f_, G_, Y_ = spec['affine_terms_fn'](x, p)
            LfV = gradV @ f_
            LgV = gradV @ G_
            LyV = gradV @ Y_
            b = -0.1 * V - LfV - LyV * adapt_st.a_hat - jnp.abs(LyV) * adapt_st.radius
            violation = jnp.dot(LgV, u_nom) - b
            denom = jnp.maximum(jnp.dot(LgV, LgV), 0.1)
            alpha = jax.nn.relu(violation) / denom
            alpha = jax.lax.stop_gradient(alpha)
            LgV_sg = jax.lax.stop_gradient(LgV)
            u = u_nom - alpha * LgV_sg
            u = jnp.clip(u, spec['u_min'], spec['u_max'])
            return jnp.sum(u**2)

        gi_sg3 = jax.grad(loss_sg_both, argnums=(0, 1))(lyap_phi, policy_params)
        has_nan_sg3 = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi_sg3))
        print(f"  stop_gradient(both alpha & LgV): NaN = {has_nan_sg3}")

        # Test: no clip_bounds
        clf_cfg_nobound = CLFConfig(enabled=True, lambda_clf=0.1, eps_proj=0.1,
                                     enforce_input_bounds=False)
        def loss_nobound(phi, pp, x=xi):
            lp = reconstruct(phi)
            obs = jnp.concatenate([spec['make_obs'](x), jnp.array([adapt_st.a_hat, adapt_st.radius])])
            u_nom = policy_apply(pp, obs, spec['u_min'], spec['u_max'], hidden, out_dim=2)
            u, _ = clf_shield(u_nom, x, adapt_st, lp, lyap_cfg, clf_cfg_nobound, p,
                              affine_terms_fn=spec['affine_terms_fn'], alpha_max=0.0)
            return jnp.sum(u**2)

        gi_nb = jax.grad(loss_nobound, argnums=(0, 1))(lyap_phi, policy_params)
        has_nan_nb = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(gi_nb))
        print(f"  No input bound clipping: NaN = {has_nan_nb}")
