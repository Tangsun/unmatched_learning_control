# Adaptive CLF Shield (JAX) - Project Guide

This project implements an adaptive robust CLF shield around a nominal policy for Acrobot stabilization.  
The codebase is now split into modular files to make reading and swapping components easier.

## Quick Start

Use your configured conda env:

```bash
conda run -n dqs_jax python -m adaptive_clf.train
```

Backward-compatible entrypoint still works:

```bash
conda run -n dqs_jax python acrobot_adaptive_clf_jax.py
```

## Code Layout

- `adaptive_clf/configs.py`: dataclasses and shared types
  - `AcrobotParams`, `LyapunovConfig`, `AdaptiveConfig`, `CLFConfig`, `RolloutConfig`
- `adaptive_clf/nn.py`: MLP and policy helpers
  - `init_policy_params`, `policy_apply`
- `adaptive_clf/acrobot.py`: dynamics, affine decomposition, RK4, linearization, LQR
  - `acrobot_affine_terms`, `rk4_step`, `solve_lqr_P`
- `adaptive_clf/lyapunov.py`: Lyapunov models and gradients
  - fixed quadratic, learned quadratic, `mlp_psd`
- `adaptive_clf/shield.py`: robust CLF terms and projection operators
  - `robust_clf_constraint_terms`, `clf_shield`
- `adaptive_clf/adaptive.py`: online adaptive update and observation builder
  - `adaptive_update_simple`, `make_policy_observation`
- `adaptive_clf/rollout.py`: rollout objective and batching
  - `episode_rollout`, `batched_rollout_loss`, `value_and_grad_loss`
- `adaptive_clf/train.py`: runnable entrypoint (single forward+backward pass)
- `acrobot_adaptive_clf_jax.py`: backward-compatible shim that re-exports package symbols

## Recommended Reading Order

If you are reviewing for understanding:

1. `adaptive_clf/train.py`  
   See end-to-end wiring first.
2. `adaptive_clf/rollout.py`  
   Understand one simulation step and the training loss.
3. `adaptive_clf/shield.py`  
   Understand robust CLF inequality and projection logic.
4. `adaptive_clf/adaptive.py`  
   Understand what adaptive state is fed to policy.
5. `adaptive_clf/acrobot.py`  
   Verify physical model and affine decomposition.
6. `adaptive_clf/lyapunov.py`  
   Compare fixed vs learned Lyapunov options.
7. `adaptive_clf/configs.py`  
   Tune hyperparameters last, after flow is clear.

## Swapping Ingredients

### Swap Lyapunov function

Change `LyapunovConfig.mode` and relevant initialization in `default_experiment_setup()`:

- `"quadratic_fixed"`: uses provided `P_init`
- `"quadratic_learned"`: trainable SPD parameterization
- `"mlp_psd"`: neural PSD form

### Swap dynamical system

Current shield logic expects control-affine terms `(f, g, y)`.  
To add a new system, mirror the Acrobot interface:

- `*_affine_terms(x, params) -> (f, g, y)`
- `*_dynamics_true(x, u, param_true, params)`
- optional linearization helper for LQR-like initialization

Then update rollout calls to use the new dynamics functions.

## About Current NaNs

The current run executes, but metrics can become `nan` during rollout.  
That usually indicates numerical instability, not necessarily architecture errors.

Recent stabilization change already applied:

- In `adaptive_clf/acrobot.py`, `acrobot_affine_terms()` now uses
  regularized linear solves (`jnp.linalg.solve`) instead of explicit matrix inversion.

### Practical debug checklist (next pass)

1. Add finite checks in rollout history (`x`, `u`, `V`, `a_hat`, `radius`) to locate first NaN timestep.
2. Clip or bound sensitive terms:
   - state magnitude before dynamics eval
   - `a_hat` update residual magnitude
   - Lyapunov values and Lie derivative intermediates
3. Reduce integration aggressiveness:
   - smaller `dt`
   - shorter horizon during debugging
4. Temporarily disable adaptation to isolate source:
   - hold `a_hat` and `radius` fixed
5. Compare shield on/off with same seed and initial batch.

## Scope Notes

- `adaptive_update_simple()` is a practical placeholder, not yet theorem-grade certified set update.
- `train.py` currently demonstrates one objective+gradient evaluation; it is not a full optimizer loop yet.

