"""Adaptive CLF-shielded learning for control-affine systems."""

# --- Config dataclasses ---
from .configs import (
    AcrobotParams,
    CartPoleParams,
    DubinsParams,
    AdaptiveConfig,
    AdaptiveState,
    Array,
    CLFConfig,
    LyapunovConfig,
    MLPConfig,
    PyTree,
    RolloutConfig,
)

# --- Neural network ---
from .nn import apply_mlp, glorot_uniform, init_mlp_params, init_policy_params, policy_apply

# --- Generic integrator ---
from .integrator import rk4_step_generic

# --- Lyapunov ---
from .lyapunov import (
    init_lyapunov_params,
    lyapunov_matrix,
    lyapunov_value,
    lyapunov_value_and_grad,
)

# --- CLF shield ---
from .shield import (
    AffineTermsFn,
    clf_shield,
    halfspace_projection,
    robust_clf_constraint_terms,
)

# --- Adaptive estimator ---
from .adaptive import adaptive_update_simple, init_adaptive_state, make_policy_observation

# --- System registry ---
from .systems import get_system, list_systems

# --- Acrobot ---
from .acrobot import (
    acrobot_affine_terms,
    acrobot_dynamics_true,
    acrobot_terms,
    linearize_acrobot_at_upright,
    rk4_step,
    solve_lqr_P,
)

# --- Cart-pole ---
from .cartpole import (
    cartpole_affine_terms,
    cartpole_dynamics,
    linearize_cartpole_at_upright,
    rk4_step_cartpole,
    solve_cartpole_lqr,
)

# --- Dubins ---
from .dubins import (
    dubins_affine_terms,
    dubins_dynamics,
    linearize_dubins,
    rk4_step_dubins,
    solve_dubins_lqr,
)

# --- Acrobot rollout (legacy) ---
from .rollout import (
    batched_rollout_loss,
    default_experiment_setup,
    episode_rollout,
    sample_local_batch,
    sample_mixed_batch,
    sample_curriculum_batch,
    sample_swingup_batch,
    sample_uniform_batch,
    stage_cost,
    value_and_grad_loss,
    value_and_grad_joint,
)
