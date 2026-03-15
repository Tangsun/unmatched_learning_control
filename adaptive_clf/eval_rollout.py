"""Compiled eval rollouts using jax.lax.scan.

Usage patterns:

  # One-off convenience (re-JITs each call -- fine for single use):
  res = compiled_eval_rollout(x0, a_true, ...)

  # Reusable (build once, call many times with different x0/a_true):
  rollout_fn = make_eval_rollout_fn(policy_params, lyap_params, ...)
  res1 = rollout_fn(x0_1, a_true_1)
  res2 = rollout_fn(x0_2, a_true_2)

  # Batched metrics (vmap over ICs, single JIT):
  norms = batched_metrics_rollout(x0s, a_true, ...)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from .configs import AdaptiveConfig, AdaptiveState, Array, CLFConfig, LyapunovConfig
from .adaptive import adaptive_update_observer, init_adaptive_state
from .integrator import rk4_step_generic
from .lyapunov import lyapunov_value_and_grad
from .nn import policy_apply
from .shield import clf_shield


def _resolve_bounds(spec, p):
    """Return (u_min, u_max) as jnp arrays."""
    if spec["ctrl_dim"] == 1:
        return jnp.atleast_1d(jnp.array(p.u_min)), jnp.atleast_1d(jnp.array(p.u_max))
    return jnp.asarray(spec["u_min"]), jnp.asarray(spec["u_max"])


def _make_adapt_st0(p, adapt_cfg, state_dim, x0, a_range=None):
    """Create initial adaptive state, handling None adapt_cfg."""
    if adapt_cfg is not None and adapt_cfg.adapt_enabled:
        return init_adaptive_state(p, adapt_cfg, state_dim=state_dim, x0=x0,
                                   a_range=a_range)
    return AdaptiveState(
        a_hat=jnp.array(0.0), info=jnp.array(1e-6),
        radius=jnp.array(0.0),
        x_hat=jnp.zeros(state_dim),
        w=jnp.zeros(state_dim),
        eta=jnp.zeros(state_dim))


def _apply_eval_shield(
    *,
    u_nom,
    x,
    adapt_st,
    lyap_params,
    lyap_cfg,
    clf_cfg,
    p,
    affine_fn,
    u_min,
    u_max,
    skip_shield_small_lgv,
):
    """Apply the CLF shield, optionally bypassing it when ||LgV||^2 is small."""
    u_shield, shield_aux = clf_shield(
        u_nom=u_nom, x=x, adaptive_state=adapt_st,
        lyap_params=lyap_params, lyap_cfg=lyap_cfg,
        clf_cfg=clf_cfg, p=p, affine_terms_fn=affine_fn,
        alpha_max=0.0,
        input_bounds=(u_min, u_max),
    )
    skip_small_lgv = jnp.logical_and(
        skip_shield_small_lgv,
        shield_aux["a_norm_sq"] < clf_cfg.eps_proj,
    )
    u_shield_vec = jnp.atleast_1d(u_shield)
    u_nom_clipped = jnp.atleast_1d(jnp.clip(u_nom, u_min, u_max))
    u = jnp.where(skip_small_lgv, u_nom_clipped, u_shield_vec).squeeze()
    return u, u_shield_vec.squeeze(), shield_aux, skip_small_lgv


# ---------------------------------------------------------------------------
# Factory: build reusable JIT-compiled rollout functions
# ---------------------------------------------------------------------------

def make_eval_rollout_fn(
    policy_params, lyap_params, lyap_cfg, lambda_clf,
    spec, hidden_sizes, adapt_cfg,
    horizon=400, dt=0.05, use_shield=True, policy_obs_dim=None,
    enforce_input_bounds=False, initial_radius=None,
    eps_proj=0.1, skip_shield_small_lgv=False,
):
    """Build a reusable compiled eval rollout function.

    Returns a callable: (x0, a_true) -> dict with numpy arrays.
    The returned function is JIT-compiled on first call and cached.

    If enforce_input_bounds=True, the shield clips its projected control
    to [u_min, u_max] after projection.
    """
    import numpy as np

    p = spec["params"]
    state_dim = spec["state_dim"]
    ctrl_dim = spec["ctrl_dim"]
    obs_dim = spec["obs_dim"]
    if policy_obs_dim is None:
        policy_obs_dim = obs_dim

    clf_cfg = CLFConfig(
        enabled=use_shield, lambda_clf=lambda_clf, eps_proj=eps_proj,
        enforce_input_bounds=enforce_input_bounds,
    )
    use_observer = adapt_cfg is not None and adapt_cfg.use_observer
    _adapt_cfg = adapt_cfg if adapt_cfg is not None else AdaptiveConfig()
    u_min, u_max = _resolve_bounds(spec, p)
    augment_obs = policy_obs_dim > obs_dim

    affine_fn = spec["affine_terms_fn"]
    dynamics_fn = spec["dynamics_fn"]
    make_obs = spec["make_obs"]
    wrap_fn = spec["wrap_state"]

    # Build the scan step (closure captures everything static)
    def step(carry, _):
        x_raw, adapt_st, a_true = carry
        x = wrap_fn(x_raw)

        obs = make_obs(x)
        if augment_obs:
            obs = jnp.concatenate([obs, jnp.array([adapt_st.a_hat, adapt_st.radius])])

        u_nom = policy_apply(policy_params, obs, u_min, u_max,
                             hidden_sizes, out_dim=ctrl_dim)

        # Always compute gradV and G for diagnostics
        V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
        f_x, g_mat, y_x = affine_fn(x, p)
        LfV = gradV @ f_x
        LgV_nom = gradV @ g_mat
        LyV = gradV @ y_x
        b_nom = (
            -clf_cfg.lambda_clf * V - LfV
            - LyV * adapt_st.a_hat
            - jnp.abs(LyV) * adapt_st.radius
        )
        nominal_violation = jnp.dot(LgV_nom, u_nom) - b_nom
        nominal_feasible = jnp.where(nominal_violation <= 0.0, 1.0, 0.0)

        if use_shield:
            u, u_shield_raw, shield_aux, skip_small_lgv = _apply_eval_shield(
                u_nom=u_nom, x=x, adapt_st=adapt_st,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_fn=affine_fn,
                u_min=u_min, u_max=u_max,
                skip_shield_small_lgv=skip_shield_small_lgv,
            )
            V = shield_aux["V"]
            feasible = jnp.where(skip_small_lgv, nominal_feasible,
                                 shield_aux["feasible"])
            LgV = shield_aux["LgV"]
        else:
            u = jnp.clip(u_nom, u_min, u_max)
            u_shield_raw = u
            LgV = gradV @ g_mat
            feasible = jnp.array(1.0)
            skip_small_lgv = jnp.array(0.0)

        u_nom_vec = jnp.atleast_1d(u_nom)
        u_applied_vec = jnp.atleast_1d(u)
        u_shield_vec = jnp.atleast_1d(u_shield_raw)
        shield_control_delta = jnp.linalg.norm(u_shield_vec - u_nom_vec)
        applied_control_delta = jnp.linalg.norm(u_applied_vec - u_nom_vec)
        shield_projected_residual = jnp.dot(LgV_nom, u_shield_vec) - b_nom
        applied_robust_residual = jnp.dot(LgV_nom, u_applied_vec) - b_nom
        true_clf_residual = (
            LfV
            + jnp.dot(LgV_nom, u_applied_vec)
            + LyV * a_true
            + clf_cfg.lambda_clf * V
        )

        if use_observer:
            adapt_st_next = adaptive_update_observer(
                adapt_st, x, u, dt, p, _adapt_cfg,
                affine_terms_fn=affine_fn,
                dynamics_fn=dynamics_fn,
                a_true=a_true,
            )
        else:
            adapt_st_next = adapt_st

        x_next = rk4_step_generic(dynamics_fn, x, u, a_true, dt, p)

        # Per-column norms of G(x): ||g_i|| for each control channel
        g_col_norms = jnp.array([jnp.linalg.norm(g_mat[:, j])
                                 for j in range(g_mat.shape[1])])

        outputs = {
            "x": x,
            "u": u,
            "u_nom": u_nom,
            "V": V,
            "a_hat": adapt_st.a_hat,
            "radius": adapt_st.radius,
            "eta_norm": jnp.linalg.norm(adapt_st.eta),
            "e_norm": jnp.linalg.norm(x - adapt_st.x_hat),
            "w_norm": jnp.linalg.norm(adapt_st.w),
            "feasible": feasible,
            "nominal_feasible": nominal_feasible,
            "nominal_violation": nominal_violation,
            "LgV": LgV,
            "gradV": gradV,
            "g_col_norms": g_col_norms,
            "shield_skip_small_lgv": skip_small_lgv.astype(jnp.float32),
            "shield_control_delta": shield_control_delta,
            "applied_control_delta": applied_control_delta,
            "shield_projected_residual": shield_projected_residual,
            "applied_robust_residual": applied_robust_residual,
            "true_clf_residual": true_clf_residual,
        }
        return (x_next, adapt_st_next, a_true), outputs

    @jax.jit
    def _rollout_jit(x0, a_true, adapt_st0):
        init_carry = (x0, adapt_st0, a_true)
        (x_final, _, _), history = jax.lax.scan(
            step, init_carry, None, length=horizon,
        )
        return x_final, history

    def rollout(x0, a_true):
        x0_jax = jnp.array(x0, dtype=jnp.float32)
        a_true_jax = jnp.asarray(a_true, dtype=jnp.float32)
        adapt_st0 = _make_adapt_st0(p, adapt_cfg, state_dim, x0_jax,
                                     a_range=initial_radius)

        x_final, history = _rollout_jit(x0_jax, a_true_jax, adapt_st0)

        # Single host transfer
        xs_mid = np.asarray(history["x"])
        x_final_np = np.asarray(x_final)
        xs = np.concatenate([xs_mid, x_final_np[None, :]], axis=0)

        return {
            "xs": xs,
            "us": np.asarray(history["u"]),
            "u_noms": np.asarray(history["u_nom"]),
            "a_hats": np.asarray(history["a_hat"]),
            "radii": np.asarray(history["radius"]),
            "etas": np.asarray(history["eta_norm"]),
            "es": np.asarray(history["e_norm"]),
            "ws": np.asarray(history["w_norm"]),
            "Vs": np.asarray(history["V"]),
            "feasibles": np.asarray(history["feasible"]),
            "nominal_feasibles": np.asarray(history["nominal_feasible"]),
            "nominal_violations": np.asarray(history["nominal_violation"]),
            "LgVs": np.asarray(history["LgV"]),
            "gradVs": np.asarray(history["gradV"]),
            "g_col_norms": np.asarray(history["g_col_norms"]),
            "shield_small_lgv_skips": np.asarray(history["shield_skip_small_lgv"]),
            "shield_control_deltas": np.asarray(history["shield_control_delta"]),
            "applied_control_deltas": np.asarray(history["applied_control_delta"]),
            "shield_projected_residuals": np.asarray(history["shield_projected_residual"]),
            "applied_robust_residuals": np.asarray(history["applied_robust_residual"]),
            "true_clf_residuals": np.asarray(history["true_clf_residual"]),
            "eps_proj": float(eps_proj),
            "skip_shield_small_lgv": bool(skip_shield_small_lgv),
            "dt": float(dt),
            "a_true": float(a_true),
        }

    return rollout


def make_metrics_rollout_fn(
    policy_params, lyap_params, lyap_cfg, lambda_clf,
    spec, hidden_sizes, adapt_cfg,
    horizon=400, dt=0.05, use_shield=True, policy_obs_dim=None,
    enforce_input_bounds=False, initial_radius=None,
    eps_proj=0.1, skip_shield_small_lgv=False,
):
    """Build a reusable compiled metrics-only rollout.

    Returns a callable: (x0, a_true) -> terminal_norm (scalar).
    """
    p = spec["params"]
    state_dim = spec["state_dim"]
    ctrl_dim = spec["ctrl_dim"]
    obs_dim = spec["obs_dim"]
    if policy_obs_dim is None:
        policy_obs_dim = obs_dim

    clf_cfg = CLFConfig(enabled=use_shield, lambda_clf=lambda_clf, eps_proj=eps_proj,
                        enforce_input_bounds=enforce_input_bounds)
    use_observer = adapt_cfg is not None and adapt_cfg.use_observer
    _adapt_cfg = adapt_cfg if adapt_cfg is not None else AdaptiveConfig()
    u_min, u_max = _resolve_bounds(spec, p)
    augment_obs = policy_obs_dim > obs_dim

    affine_fn = spec["affine_terms_fn"]
    dynamics_fn = spec["dynamics_fn"]
    make_obs = spec["make_obs"]
    wrap_fn = spec["wrap_state"]

    def step(carry, _):
        x_raw, adapt_st, a_true = carry
        x = wrap_fn(x_raw)

        obs = make_obs(x)
        if augment_obs:
            obs = jnp.concatenate([obs, jnp.array([adapt_st.a_hat, adapt_st.radius])])

        u_nom = policy_apply(policy_params, obs, u_min, u_max,
                             hidden_sizes, out_dim=ctrl_dim)

        if use_shield:
            u, _, _ = _apply_eval_shield(
                u_nom=u_nom, x=x, adapt_st=adapt_st,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_fn=affine_fn,
                u_min=u_min, u_max=u_max,
                skip_shield_small_lgv=skip_shield_small_lgv,
            )
        else:
            u = jnp.clip(u_nom, u_min, u_max)

        if use_observer:
            adapt_st_next = adaptive_update_observer(
                adapt_st, x, u, dt, p, _adapt_cfg,
                affine_terms_fn=affine_fn,
                dynamics_fn=dynamics_fn,
                a_true=a_true,
            )
        else:
            adapt_st_next = adapt_st

        x_next = rk4_step_generic(dynamics_fn, x, u, a_true, dt, p)
        return (x_next, adapt_st_next, a_true), None

    @jax.jit
    def _rollout_jit(x0, a_true, adapt_st0):
        init_carry = (x0, adapt_st0, a_true)
        (x_final, _, _), _ = jax.lax.scan(
            step, init_carry, None, length=horizon,
        )
        return jnp.linalg.norm(x_final)

    def rollout(x0, a_true):
        x0_jax = jnp.array(x0, dtype=jnp.float32)
        a_true_jax = jnp.asarray(a_true, dtype=jnp.float32)
        adapt_st0 = _make_adapt_st0(p, adapt_cfg, state_dim, x0_jax,
                                     a_range=initial_radius)
        return float(_rollout_jit(x0_jax, a_true_jax, adapt_st0))

    return rollout


# ---------------------------------------------------------------------------
# Convenience wrappers (re-build on each call -- for one-off use)
# ---------------------------------------------------------------------------

def compiled_eval_rollout(
    x0, a_true, policy_params, lyap_params, lyap_cfg, lambda_clf,
    spec, hidden_sizes, adapt_cfg, horizon=400, dt=0.05,
    use_shield=True, policy_obs_dim=None, enforce_input_bounds=False,
    initial_radius=None, eps_proj=0.1, skip_shield_small_lgv=False,
):
    """One-off compiled eval rollout. Prefer make_eval_rollout_fn for repeated calls."""
    fn = make_eval_rollout_fn(
        policy_params, lyap_params, lyap_cfg, lambda_clf,
        spec, hidden_sizes, adapt_cfg,
        horizon, dt, use_shield, policy_obs_dim,
        enforce_input_bounds=enforce_input_bounds,
        initial_radius=initial_radius,
        eps_proj=eps_proj,
        skip_shield_small_lgv=skip_shield_small_lgv,
    )
    return fn(x0, a_true)


def metrics_only_rollout(
    x0, a_true, policy_params, lyap_params, lyap_cfg, lambda_clf,
    spec, hidden_sizes, adapt_cfg, horizon=400, dt=0.05,
    use_shield=True, policy_obs_dim=None, enforce_input_bounds=False,
    initial_radius=None, eps_proj=0.1, skip_shield_small_lgv=False,
):
    """One-off metrics-only rollout. Prefer make_metrics_rollout_fn for repeated calls."""
    fn = make_metrics_rollout_fn(
        policy_params, lyap_params, lyap_cfg, lambda_clf,
        spec, hidden_sizes, adapt_cfg,
        horizon, dt, use_shield, policy_obs_dim,
        enforce_input_bounds=enforce_input_bounds,
        initial_radius=initial_radius,
        eps_proj=eps_proj,
        skip_shield_small_lgv=skip_shield_small_lgv,
    )
    return fn(x0, a_true)


# ---------------------------------------------------------------------------
# Batched (vmap) metrics rollout
# ---------------------------------------------------------------------------

def make_batched_metrics_fn(
    policy_params, lyap_params, lyap_cfg, lambda_clf,
    spec, hidden_sizes, adapt_cfg,
    horizon=400, dt=0.05, use_shield=True, policy_obs_dim=None,
    enforce_input_bounds=False, initial_radius=None,
    eps_proj=0.1, skip_shield_small_lgv=False,
):
    """Build a reusable vmap'd metrics rollout.

    Returns a callable: (x0s, a_true) -> numpy array of terminal norms.
    JIT-compiled on first call, cached for subsequent calls.
    """
    import numpy as np

    p = spec["params"]
    state_dim = spec["state_dim"]
    ctrl_dim = spec["ctrl_dim"]
    obs_dim = spec["obs_dim"]
    if policy_obs_dim is None:
        policy_obs_dim = obs_dim

    clf_cfg = CLFConfig(enabled=use_shield, lambda_clf=lambda_clf, eps_proj=eps_proj,
                        enforce_input_bounds=enforce_input_bounds)
    use_observer = adapt_cfg is not None and adapt_cfg.use_observer
    _adapt_cfg = adapt_cfg if adapt_cfg is not None else AdaptiveConfig()
    u_min, u_max = _resolve_bounds(spec, p)
    augment_obs = policy_obs_dim > obs_dim

    affine_fn = spec["affine_terms_fn"]
    dynamics_fn = spec["dynamics_fn"]
    make_obs = spec["make_obs"]
    wrap_fn = spec["wrap_state"]

    def step(carry, _):
        x_raw, adapt_st, a_true_c = carry
        x = wrap_fn(x_raw)

        obs = make_obs(x)
        if augment_obs:
            obs = jnp.concatenate([obs, jnp.array([adapt_st.a_hat, adapt_st.radius])])

        u_nom = policy_apply(policy_params, obs, u_min, u_max,
                             hidden_sizes, out_dim=ctrl_dim)

        if use_shield:
            u, _, _ = _apply_eval_shield(
                u_nom=u_nom, x=x, adapt_st=adapt_st,
                lyap_params=lyap_params, lyap_cfg=lyap_cfg,
                clf_cfg=clf_cfg, p=p, affine_fn=affine_fn,
                u_min=u_min, u_max=u_max,
                skip_shield_small_lgv=skip_shield_small_lgv,
            )
        else:
            u = jnp.clip(u_nom, u_min, u_max)

        if use_observer:
            adapt_st_next = adaptive_update_observer(
                adapt_st, x, u, dt, p, _adapt_cfg,
                affine_terms_fn=affine_fn,
                dynamics_fn=dynamics_fn,
                a_true=a_true_c,
            )
        else:
            adapt_st_next = adapt_st

        x_next = rk4_step_generic(dynamics_fn, x, u, a_true_c, dt, p)
        return (x_next, adapt_st_next, a_true_c), None

    def single_rollout(x0, a_true_single, adapt_st0):
        init_carry = (x0, adapt_st0, a_true_single)
        (x_final, _, _), _ = jax.lax.scan(
            step, init_carry, None, length=horizon,
        )
        return jnp.linalg.norm(x_final)

    @jax.jit
    def _batched(x0s_b, a_b, ast_b):
        return jax.vmap(single_rollout)(x0s_b, a_b, ast_b)

    def batched_rollout(x0s, a_true):
        x0s_jax = jnp.array(x0s, dtype=jnp.float32)
        a_true_jax = jnp.asarray(a_true, dtype=jnp.float32)

        adapt_st0_batch = jax.vmap(
            lambda x0: _make_adapt_st0(p, adapt_cfg, state_dim, x0,
                                       a_range=initial_radius)
        )(x0s_jax)

        a_batch = jnp.broadcast_to(a_true_jax, (x0s_jax.shape[0],))
        norms = _batched(x0s_jax, a_batch, adapt_st0_batch)
        return np.asarray(norms)

    return batched_rollout


def batched_metrics_rollout(
    x0s, a_true, policy_params, lyap_params, lyap_cfg, lambda_clf,
    spec, hidden_sizes, adapt_cfg, horizon=400, dt=0.05,
    use_shield=True, policy_obs_dim=None, enforce_input_bounds=False,
    initial_radius=None, eps_proj=0.1, skip_shield_small_lgv=False,
):
    """One-off batched metrics rollout. Prefer make_batched_metrics_fn for repeated calls."""
    fn = make_batched_metrics_fn(
        policy_params, lyap_params, lyap_cfg, lambda_clf,
        spec, hidden_sizes, adapt_cfg,
        horizon, dt, use_shield, policy_obs_dim,
        enforce_input_bounds=enforce_input_bounds,
        initial_radius=initial_radius,
        eps_proj=eps_proj,
        skip_shield_small_lgv=skip_shield_small_lgv,
    )
    return fn(x0s, a_true)
