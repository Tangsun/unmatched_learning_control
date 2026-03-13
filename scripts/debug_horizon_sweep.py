"""Test which horizon + eps_proj combinations give clean gradients for PVTOL shield-diff."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import jax
import jax.numpy as jnp
import pickle

from adaptive_clf.configs import AdaptiveConfig, CLFConfig
from adaptive_clf.train_unified import CostConfig
from adaptive_clf.systems import get_system
from adaptive_clf.train_lyapunov import _batched_loss_lyap

with open('runs/pvtol_p3_observer_scratch/policy_params.pkl', 'rb') as f:
    d = pickle.load(f)

spec = get_system('pvtol')
lyap_cfg = d['lyap_cfg']
lyap_params = d['lyap_params']
hidden = d['hidden_sizes']
policy_params = d['nn']

cost_cfg = CostConfig(
    state_weights=tuple(spec.get("cost_weights", {}).get("state_weights", [1.0]*6)),
    u_weight=spec.get("cost_weights", {}).get("u_weight", 0.01),
    terminal_scale=spec.get("cost_weights", {}).get("terminal_scale", 10.0),
    proj_weight=0.1,
)
adapt_cfg = AdaptiveConfig(
    adapt_enabled=True, eta=0.02,
    use_observer=True, observer_k=3.0, observer_gamma=20.0,
)

lyap_static = {k: v for k, v in lyap_params.items() if k in ("mode", "x_eq")}
lyap_phi = lyap_params["phi"]
lqr_K = d.get("lqr_K", jnp.zeros((2, 6)))

key = jax.random.PRNGKey(0)
batch_x0, _ = spec['sample_ics'](key, 64, 0.5, 0.0)
batch_a = jax.random.uniform(jax.random.PRNGKey(1), (64,), minval=-1.5, maxval=1.5)
all_params = {"policy": policy_params, "lyap_phi": lyap_phi}

print(f"{'horizon':>8s}  {'eps_proj':>8s}  {'NaN?':>5s}  {'grad_norm':>12s}  {'loss':>10s}")
print("-" * 55)

for horizon in [10, 25, 50, 75, 100, 150, 200]:
    for eps_proj in [0.1]:
        clf_cfg = CLFConfig(enabled=True, lambda_clf=0.1, eps_proj=eps_proj)
        def loss_fn(params, h=horizon, cc=clf_cfg):
            phi = params["lyap_phi"]
            lp = {**lyap_static, "phi": phi}
            return _batched_loss_lyap(
                params["policy"], lp, batch_x0, batch_a,
                spec, hidden, lqr_K, h, 0.02,
                cost_cfg, lyap_cfg, cc,
                True, 0.1, jnp.float32(10.0), "nn_only",
                adapt_cfg=adapt_cfg, shield_diff=True,
                alpha_max=0.0,
            )
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(all_params)
        grad_norm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(grads)))
        has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(grads))
        status = "YES" if has_nan else "no"
        print(f"{horizon:8d}  {eps_proj:8.1f}  {status:>5s}  {float(grad_norm):12.2f}  {float(loss):10.3f}")
