"""Debug: replicate exact training loss to find NaN source."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import jax
import jax.numpy as jnp
import pickle

from adaptive_clf.configs import (
    AdaptiveConfig, AdaptiveState, CLFConfig, LyapunovConfig,
)
from adaptive_clf.train_unified import CostConfig
from adaptive_clf.systems import get_system
from adaptive_clf.train_lyapunov import _batched_loss_lyap

with open('runs/pvtol_p3_observer_scratch/policy_params.pkl', 'rb') as f:
    d = pickle.load(f)

spec = get_system('pvtol')
p = spec['params']
lyap_cfg = d['lyap_cfg']
lyap_params = d['lyap_params']
hidden = d['hidden_sizes']
policy_params = d['nn']

# Match training config
clf_cfg = CLFConfig(enabled=True, lambda_clf=0.1, eps_proj=0.1)
cost_cfg = CostConfig(
    state_weights=tuple(spec.get("cost_weights", {}).get("state_weights", [1.0]*6)),
    u_weight=spec.get("cost_weights", {}).get("u_weight", 0.01),
    terminal_scale=spec.get("cost_weights", {}).get("terminal_scale", 10.0),
    proj_weight=0.1,
    infeasible_weight=0.0,
)

adapt_cfg = AdaptiveConfig(
    adapt_enabled=True, eta=0.02,
    use_observer=True, observer_k=3.0, observer_gamma=20.0,
)

lyap_static = {k: v for k, v in lyap_params.items() if k in ("mode", "x_eq")}
lyap_phi = lyap_params["phi"]

lqr_K = d.get("lqr_K", jnp.zeros((2, 6)))

# Sample a batch like training
key = jax.random.PRNGKey(0)
batch_x0, batch_a = spec['sample_ics'](key, 64, 0.5, 0.0)
# Randomize a
key2 = jax.random.PRNGKey(1)
batch_a = jax.random.uniform(key2, (64,), minval=-1.5, maxval=1.5)

all_params = {"policy": policy_params, "lyap_phi": lyap_phi}

def loss_fn(params, alpha_max_val=0.0):
    phi = params["lyap_phi"]
    lp = {**lyap_static, "phi": phi}
    return _batched_loss_lyap(
        params["policy"], lp, batch_x0, batch_a,
        spec, hidden, lqr_K, 200, 0.02,
        cost_cfg, lyap_cfg, clf_cfg,
        True, 0.1, jnp.float32(10.0), "nn_only",
        adapt_cfg=adapt_cfg, shield_diff=True,
        alpha_max=alpha_max_val,
    )

# Test 1: alpha_max=0.0
print("Test 1: Full training loss, alpha_max=0.0")
(loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(all_params)
grad_norm = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(grads)))
has_nan = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(grads))
print(f"  loss={float(loss):.4f}, grad_norm={float(grad_norm):.4f}, NaN={has_nan}")
print(f"  terminal_norm={float(metrics['terminal_norm']):.4f}")
print(f"  mean_V={float(metrics['mean_V']):.4f}")
print(f"  mean_feasible={float(metrics['mean_feasible']):.4f}")

# Test 2: alpha_max=10.0
print("\nTest 2: Full training loss, alpha_max=10.0")
def loss_fn2(params):
    return loss_fn(params, alpha_max_val=10.0)
(loss2, metrics2), grads2 = jax.value_and_grad(loss_fn2, has_aux=True)(all_params)
grad_norm2 = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(grads2)))
has_nan2 = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(grads2))
print(f"  loss={float(loss2):.4f}, grad_norm={float(grad_norm2):.4f}, NaN={has_nan2}")

# Test 3: alpha_max=1.0
print("\nTest 3: Full training loss, alpha_max=1.0")
def loss_fn3(params):
    return loss_fn(params, alpha_max_val=1.0)
(loss3, metrics3), grads3 = jax.value_and_grad(loss_fn3, has_aux=True)(all_params)
grad_norm3 = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(grads3)))
has_nan3 = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(grads3))
print(f"  loss={float(loss3):.4f}, grad_norm={float(grad_norm3):.4f}, NaN={has_nan3}")

# Test 4: no proj_cost (proj_weight=0)
print("\nTest 4: Full training loss, alpha_max=0, proj_weight=0")
cost_cfg_noproj = CostConfig(
    state_weights=cost_cfg.state_weights,
    u_weight=cost_cfg.u_weight,
    terminal_scale=cost_cfg.terminal_scale,
    proj_weight=0.0,
    infeasible_weight=0.0,
)
def loss_fn4(params):
    phi = params["lyap_phi"]
    lp = {**lyap_static, "phi": phi}
    return _batched_loss_lyap(
        params["policy"], lp, batch_x0, batch_a,
        spec, hidden, lqr_K, 200, 0.02,
        cost_cfg_noproj, lyap_cfg, clf_cfg,
        True, 0.1, jnp.float32(10.0), "nn_only",
        adapt_cfg=adapt_cfg, shield_diff=True,
        alpha_max=0.0,
    )
(loss4, metrics4), grads4 = jax.value_and_grad(loss_fn4, has_aux=True)(all_params)
grad_norm4 = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(grads4)))
has_nan4 = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(grads4))
print(f"  loss={float(loss4):.4f}, grad_norm={float(grad_norm4):.4f}, NaN={has_nan4}")

# Test 5: shorter horizon
print("\nTest 5: Shorter horizon=50, alpha_max=0")
def loss_fn5(params):
    phi = params["lyap_phi"]
    lp = {**lyap_static, "phi": phi}
    return _batched_loss_lyap(
        params["policy"], lp, batch_x0, batch_a,
        spec, hidden, lqr_K, 50, 0.02,
        cost_cfg, lyap_cfg, clf_cfg,
        True, 0.1, jnp.float32(10.0), "nn_only",
        adapt_cfg=adapt_cfg, shield_diff=True,
        alpha_max=0.0,
    )
(loss5, metrics5), grads5 = jax.value_and_grad(loss_fn5, has_aux=True)(all_params)
grad_norm5 = jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(grads5)))
has_nan5 = any(bool(jnp.any(jnp.isnan(v))) for v in jax.tree.leaves(grads5))
print(f"  loss={float(loss5):.4f}, grad_norm={float(grad_norm5):.4f}, NaN={has_nan5}")

print("\nDone.")
