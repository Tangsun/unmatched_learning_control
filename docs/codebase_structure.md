# Codebase Structure

Last updated: 2026-03-08

## Overview

JAX-based framework for learning adaptive control policies with Control Lyapunov Function (CLF) stability certificates. Supports multiple dynamical systems through a unified interface.

All systems follow the **control-affine** form:

```
xdot = f(x) + g(x) * u + y(x) * a_true
```

where `a_true` is an unknown constant uncertainty parameter.

## Directory Layout

```
adaptive_clf/                  # Core library
    configs.py                 # All dataclasses and type aliases
    systems.py                 # System registry (get_system, list_systems)
    integrator.py              # Generic RK4 integrator
    nn.py                      # MLP policy network (single- and multi-output)
    lyapunov.py                # Lyapunov function modes
    shield.py                  # CLF half-space projection
    adaptive.py                # Online parameter estimator
    acrobot.py                 # Acrobot dynamics
    cartpole.py                # Cart-pole dynamics
    dubins.py                  # Dubins car dynamics
    train_unified.py           # Unified training script (preferred)
    train_cartpole.py          # Legacy cart-pole training
    train_dubins.py            # Legacy dubins training
    train.py                   # Legacy acrobot training
    rollout.py                 # Legacy acrobot rollout/loss

scripts/                       # Evaluation and visualization
    animate_cartpole.py        # Cart-pole GIF animations
    visualize_acrobot.py       # Acrobot GIF animations
    visualize_cartpole.py      # Hybrid LQR+NN analysis plots
    reeval_and_visualize_cartpole.py  # Re-eval saved policies, comparison plots
    eval_swingup.py            # Acrobot eval
    run_swingup_experiments.py # Acrobot batch experiments

docs/                          # Experiment reports
runs/                          # Saved experiment outputs (params, plots, configs)
```

## Core Modules

### configs.py (137 lines)

All parameter and configuration dataclasses:

| Dataclass | Purpose |
|-----------|---------|
| `AcrobotParams` | Acrobot physical params (masses, lengths, inertias, control bounds) |
| `CartPoleParams` | Cart-pole params (cart/pole mass, length, control bounds) |
| `DubinsParams` | Dubins car params (reference speed, control bounds) |
| `MLPConfig` | Neural network architecture (dims, activations, output scale) |
| `LyapunovConfig` | Lyapunov mode selection and init |
| `CLFConfig` | CLF shield settings (lambda, projection epsilon) |
| `AdaptiveConfig` | Adaptive estimator settings (learning rate, radius) |
| `RolloutConfig` | Legacy acrobot training config |
| `AdaptiveState` | NamedTuple: (a_hat, info, radius) |

Type aliases: `Array = jnp.ndarray`, `PyTree = Any`

### systems.py (215 lines)

System registry mapping names to dynamics and defaults.

```python
from adaptive_clf.systems import get_system, list_systems

spec = get_system("cartpole")  # or "acrobot", "dubins"
```

Each system spec is a dict containing:

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | System identifier |
| `state_dim` | int | State dimension |
| `ctrl_dim` | int | Control dimension (1 for acrobot/cartpole, 2 for dubins) |
| `obs_dim` | int | Observation dimension (may differ from state_dim) |
| `params` | dataclass | Default physical parameters |
| `affine_terms_fn` | callable | `(x, p) -> (f, g, y)` |
| `dynamics_fn` | callable | `(x, u, a, p) -> xdot` |
| `make_obs` | callable | `(x) -> obs` (may use sin/cos encoding) |
| `wrap_state` | callable | `(x) -> x` (angle wrapping) |
| `x_eq` | tuple | Equilibrium state |
| `default_lqr_Q` | array | Default LQR Q matrix |
| `default_lqr_R` | array | Default LQR R matrix |
| `solve_lqr` | callable | `(p, Q, R, a_nom) -> (P, K)` |
| `sample_ics` | callable | `(key, batch_size, region_scale, a_true) -> (x0, a_batch)` |
| `default_cost_weights` | dict | Default cost configuration |

Dubins additionally has `u_eq`, `u_min`, `u_max` as arrays (for multi-channel bounds).

### integrator.py (51 lines)

```python
from adaptive_clf.integrator import rk4_step_generic

x_next = rk4_step_generic(dynamics_fn, x, u, a_true, dt, p, n_substeps=1)
```

Generic RK4 that works with any `dynamics_fn(x, u, a, p) -> xdot`. Supports multi-step subdivision via `n_substeps` for stiff systems.

### nn.py (98 lines)

Neural network utilities:

- `init_mlp_params(key, cfg)` -- initialize MLP from `MLPConfig`
- `apply_mlp(params, x, cfg)` -- forward pass
- `init_policy_params(key, obs_dim, hidden_sizes, out_dim=1)` -- convenience for policies
- `policy_apply(params, obs, u_min, u_max, hidden_sizes, out_dim=1)` -- policy with tanh-scaled output mapping

Multi-output support: `out_dim > 1` for systems like dubins (2D control). Uses per-channel scaling: `u = u_mid + u_half * tanh(raw)`.

### lyapunov.py (114 lines)

Three Lyapunov function modes:

| Mode | V(x) | Params |
|------|-------|--------|
| `quadratic_fixed` | `(x-x_eq)^T P (x-x_eq)` | Fixed P (e.g., from LQR) |
| `quadratic_learned` | `(x-x_eq)^T P(theta) (x-x_eq)` | Cholesky-parameterized P = LL^T |
| `mlp_psd` | `0.5 * (phi(x) - phi(0))^2 + eps * ||x||^2` | Neural network phi |

Key functions:
- `init_lyapunov_params(key, cfg)` -- mode-specific initialization
- `lyapunov_value(params, cfg, x)` -- scalar V(x)
- `lyapunov_value_and_grad(params, cfg, x)` -- (V, grad_V) tuple
- `lyapunov_matrix(params, cfg)` -- P matrix (quadratic modes only)

### shield.py (140 lines)

CLF projection operators:

- `halfspace_projection(u_nom, a_vec, b_scalar)` -- project onto `{u : a^T u <= b}`
- `robust_clf_constraint_terms(x, a_hat, radius, ...)` -- compute CLF constraint RHS
- `clf_shield(u_nom, x, adaptive_state, ...)` -- main shield: project u_nom onto CLF-safe set

All functions require `affine_terms_fn` parameter (no implicit defaults).

### adaptive.py (66 lines)

Online parameter estimator:

- `init_adaptive_state(p, adapt_cfg)` -- initialize (a_hat, info, radius)
- `adaptive_update_simple(state, x, u, a_true, dt, p, ...)` -- gradient-based update using simulator residuals
- `make_policy_observation(x, adaptive_state, adapt_cfg)` -- augment state with adaptive info

Requires `affine_terms_fn` and `dynamics_fn` parameters explicitly.

## System Dynamics Files

### acrobot.py (166 lines)

2-link underactuated pendulum. State: `[dq1, dq2, dq1_dot, dq2_dot]` (deviations from upright).
- Uncertainty: shoulder damping
- Functions: `acrobot_affine_terms`, `acrobot_dynamics_true`, `acrobot_terms` (M, C, tau_g), `rk4_step`, `linearize_acrobot_at_upright`, `solve_lqr_P`

### cartpole.py (105 lines)

Cart-pole on track. State: `[x_cart, theta, x_cart_dot, theta_dot]` (theta=0 upright).
- Uncertainty: cart friction
- Functions: `cartpole_affine_terms`, `cartpole_dynamics`, `rk4_step_cartpole`, `linearize_cartpole_at_upright`, `solve_cartpole_lqr`

### dubins.py (109 lines)

Dubins car path-following error dynamics. State: `[e_x, e_y, e_theta]`. Control: `[v, omega]`.
- Uncertainty: side-slip velocity (unmatched -- enters e_y but control only enters e_theta)
- Functions: `dubins_affine_terms`, `dubins_dynamics`, `rk4_step_dubins`, `linearize_dubins`, `solve_dubins_lqr`

## Training

### Unified script (preferred): train_unified.py (625 lines)

```bash
# NN-only baseline
python -m adaptive_clf.train_unified --system cartpole --epochs 100

# CLF with fixed Lyapunov
python -m adaptive_clf.train_unified --system cartpole --clf --region-scale 0.3

# CLF with learned Lyapunov
python -m adaptive_clf.train_unified --system cartpole --learn-lyap --region-scale 0.3

# Dubins with multi-output policy
python -m adaptive_clf.train_unified --system dubins --epochs 200

# All options
python -m adaptive_clf.train_unified \
    --system cartpole \
    --policy-mode clf \
    --epochs 100 --lr 5e-4 --batch-size 64 --pool-size 2048 \
    --horizon 200 --dt 0.02 \
    --clf --lambda-clf 0.5 --lyap-mode quadratic_fixed \
    --region-scale 0.3 --a-true 0.0 \
    --save-dir runs/my_experiment --seed 0
```

Policy modes:
- `nn_only`: pure NN policy
- `hybrid`: LQR+NN blend (smooth sigmoid based on distance from equilibrium)
- `clf`: NN + CLF shield projection

Outputs per run:
- `policy_params.pkl`: saved policy, LQR gain, Lyapunov params
- `run_config.pkl`: full experiment configuration
- `history.pkl`: per-step training metrics
- `training_curve.png`: loss, terminal norm, feasibility plots

### Legacy scripts

- `train_cartpole.py` (855 lines): cart-pole specific, includes eval and plotting
- `train_dubins.py` (805 lines): dubins specific, local multi-output policy
- `train.py` (615 lines): acrobot specific, multi-phase curriculum
- `rollout.py` (425 lines): acrobot-specific episode rollout and sampling

These still work and may have system-specific features not yet in the unified script (e.g., cart-pole energy cost, acrobot multi-phase curriculum).

## Visualization Scripts

| Script | System | Output |
|--------|--------|--------|
| `animate_cartpole.py` | cart-pole | GIF animations with cart, pole, control subplot |
| `visualize_acrobot.py` | acrobot | GIF animations of 2-link arm |
| `visualize_cartpole.py` | cart-pole | Multi-panel analysis (phase portrait, blending, decomposition) |
| `reeval_and_visualize_cartpole.py` | cart-pole | Re-eval saved runs, cross-method comparison plots, Lyapunov analysis |

## Adding a New System

1. Create `adaptive_clf/newsystem.py` with:
   - Parameter dataclass (add to `configs.py`)
   - `newsystem_affine_terms(x, p) -> (f, g, y)`
   - `newsystem_dynamics(x, u, a, p) -> xdot`
   - `linearize_newsystem(p, a_nom) -> (A, B)`
   - `solve_newsystem_lqr(p, Q, R, a_nom) -> (P, K)`

2. Register in `systems.py`:
   - Add `_newsystem_spec()` function with make_obs, wrap_state, sample_ics, defaults
   - Add to `_REGISTRY`

3. Train: `python -m adaptive_clf.train_unified --system newsystem`

## Environment

- Python 3.10, JAX with GPU (CUDA)
- Conda environment: `dqs_jax`
- GPU: NVIDIA RTX 5080
