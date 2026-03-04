from __future__ import annotations

"""Backward-compatible shim for the modular Adaptive CLF package.

This module keeps the original import surface available while delegating all
implementation to the `adaptive_clf` package.
"""

from adaptive_clf import (
    AcrobotParams,
    AdaptiveConfig,
    AdaptiveState,
    Array,
    CLFConfig,
    LyapunovConfig,
    MLPConfig,
    PyTree,
    RolloutConfig,
    acrobot_affine_terms,
    acrobot_dynamics_true,
    acrobot_terms,
    adaptive_update_simple,
    apply_mlp,
    batched_rollout_loss,
    clf_shield,
    default_experiment_setup,
    episode_rollout,
    glorot_uniform,
    halfspace_projection,
    init_adaptive_state,
    init_lyapunov_params,
    init_mlp_params,
    init_policy_params,
    linearize_acrobot_at_upright,
    lyapunov_matrix,
    lyapunov_value,
    lyapunov_value_and_grad,
    make_policy_observation,
    policy_apply,
    rk4_step,
    robust_clf_constraint_terms,
    sample_local_batch,
    solve_lqr_P,
    stage_cost,
    value_and_grad_loss,
)
from adaptive_clf.train import main


if __name__ == "__main__":  # pragma: no cover
    main()
