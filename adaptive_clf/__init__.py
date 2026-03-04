"""Adaptive CLF-shielded learning for control-affine systems."""

from .configs import (
    AcrobotParams,
    AdaptiveConfig,
    AdaptiveState,
    Array,
    CLFConfig,
    LyapunovConfig,
    MLPConfig,
    PyTree,
    RolloutConfig,
)
from .nn import apply_mlp, glorot_uniform, init_mlp_params, init_policy_params, policy_apply
from .acrobot import (
    acrobot_affine_terms,
    acrobot_dynamics_true,
    acrobot_terms,
    linearize_acrobot_at_upright,
    rk4_step,
    solve_lqr_P,
)
from .lyapunov import (
    init_lyapunov_params,
    lyapunov_matrix,
    lyapunov_value,
    lyapunov_value_and_grad,
)
from .shield import (
    clf_shield,
    halfspace_projection,
    robust_clf_constraint_terms,
)
from .adaptive import adaptive_update_simple, init_adaptive_state, make_policy_observation
from .rollout import (
    batched_rollout_loss,
    default_experiment_setup,
    episode_rollout,
    sample_local_batch,
    sample_curriculum_batch,
    sample_swingup_batch,
    stage_cost,
    value_and_grad_loss,
)

__all__ = [
    "AcrobotParams",
    "AdaptiveConfig",
    "AdaptiveState",
    "Array",
    "CLFConfig",
    "LyapunovConfig",
    "MLPConfig",
    "PyTree",
    "RolloutConfig",
    "apply_mlp",
    "glorot_uniform",
    "init_mlp_params",
    "init_policy_params",
    "policy_apply",
    "acrobot_affine_terms",
    "acrobot_dynamics_true",
    "acrobot_terms",
    "linearize_acrobot_at_upright",
    "rk4_step",
    "solve_lqr_P",
    "init_lyapunov_params",
    "lyapunov_matrix",
    "lyapunov_value",
    "lyapunov_value_and_grad",
    "clf_shield",
    "halfspace_projection",
    "robust_clf_constraint_terms",
    "adaptive_update_simple",
    "init_adaptive_state",
    "make_policy_observation",
    "batched_rollout_loss",
    "default_experiment_setup",
    "episode_rollout",
    "sample_local_batch",
    "sample_curriculum_batch",
    "sample_swingup_batch",
    "stage_cost",
    "value_and_grad_loss",
]
