"""Projection operators and CLF shield composition."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from .configs import AdaptiveState, Array, CLFConfig, LyapunovConfig
from .lyapunov import lyapunov_value_and_grad

# Type alias for control-affine dynamics: (x, params) -> (f, g, y)
AffineTermsFn = Callable[..., Tuple[Array, Array, Array]]


def halfspace_projection(
    u_nom: Array,
    a_vec: Array,
    b_scalar: Array,
    eps: float = 1e-2,
    clip_bounds: Optional[Tuple[float, float]] = None,
    alpha_max: float = 0.0,
) -> Tuple[Array, Dict[str, Array]]:
    """Project onto a CLF halfspace, optionally using a box-constrained best effort.

    Parameters
    ----------
    u_nom : nominal control input (any dimension).
    a_vec : constraint normal (same shape as u_nom).
    b_scalar : constraint right-hand side.
    eps : regularization for near-zero norm of a_vec.
    clip_bounds : if provided, (u_min, u_max) box. The nominal action is first
        clipped into the box. If it violates the halfspace, the controller moves
        toward the box point that minimizes a^T u. If the halfspace is reachable
        inside the box, a line search returns the closest point on that segment
        that satisfies the constraint. Otherwise the box-optimal point is used
        as a bounded best effort.
    alpha_max : if > 0, cap the projection gain to prevent gradient explosion
        when ||a_vec||^2 is small (infeasible states).
    """
    a_vec = jnp.atleast_1d(a_vec)
    u_nom = jnp.atleast_1d(u_nom)
    a_norm_sq = jnp.dot(a_vec, a_vec)

    if clip_bounds is not None:
        u_min = jnp.asarray(clip_bounds[0], dtype=u_nom.dtype)
        u_max = jnp.asarray(clip_bounds[1], dtype=u_nom.dtype)
        u_clipped = jnp.clip(u_nom, u_min, u_max)
        violation = jnp.dot(a_vec, u_clipped) - b_scalar

        # Best point in the box for minimizing the halfspace residual a^T u - b.
        u_box_best = jnp.where(
            a_vec > 0.0, u_min,
            jnp.where(a_vec < 0.0, u_max, u_clipped),
        )
        best_violation = jnp.dot(a_vec, u_box_best) - b_scalar
        box_feasible = best_violation <= 0.0

        # If the halfspace is reachable in the box, move along the segment from
        # clipped nominal to the box-optimal point until the residual hits zero.
        denom = jnp.maximum(violation - best_violation, 1e-12)
        t_hit = jnp.clip(violation / denom, 0.0, 1.0)
        u_line = u_clipped + t_hit * (u_box_best - u_clipped)

        use_nominal = violation <= 0.0
        use_line = jnp.logical_and(~use_nominal, box_feasible)
        u_proj = jnp.where(
            use_nominal,
            u_clipped,
            jnp.where(use_line, u_line, u_box_best),
        )
        final_violation = jnp.dot(a_vec, u_proj) - b_scalar
        used_box_best_effort = jnp.logical_and(~use_nominal, ~box_feasible)
        projection_gain = jnp.where(
            use_nominal,
            0.0,
            jnp.where(use_line, t_hit, 1.0),
        )
        aux = {
            "violation": violation,
            "final_violation": final_violation,
            "projection_gain": projection_gain,
            "a_norm_sq": a_norm_sq,
            "box_best_violation": best_violation,
            "constraint_satisfied": jnp.where(final_violation <= 1e-8, 1.0, 0.0),
            "nominal_feasible": jnp.where(violation <= 0.0, 1.0, 0.0),
            "box_feasible": jnp.where(box_feasible, 1.0, 0.0),
            "used_box_best_effort": jnp.where(used_box_best_effort, 1.0, 0.0),
        }
        return u_proj.squeeze(), aux

    violation = jnp.dot(a_vec, u_nom) - b_scalar
    denom = jnp.maximum(a_norm_sq, eps)
    alpha = jax.nn.relu(violation) / denom
    if alpha_max > 0:
        alpha = jnp.minimum(alpha, alpha_max)
    u_proj = u_nom - alpha * a_vec
    final_violation = jnp.dot(a_vec, u_proj) - b_scalar

    aux = {
        "violation": violation,
        "final_violation": final_violation,
        "projection_gain": alpha,
        "a_norm_sq": a_norm_sq,
        "constraint_satisfied": jnp.where(final_violation <= 1e-8, 1.0, 0.0),
        "nominal_feasible": jnp.where(violation <= 0.0, 1.0, 0.0),
        "box_feasible": jnp.array(1.0),
        "used_box_best_effort": jnp.array(0.0),
    }
    return u_proj.squeeze(), aux


def robust_clf_constraint_terms(x: Array,
                                a_hat: Array,
                                radius: Array,
                                lyap_params: Dict[str, Any],
                                lyap_cfg: LyapunovConfig,
                                clf_cfg: CLFConfig,
                                p: Any,
                                affine_terms_fn: AffineTermsFn = None,
                                ) -> Dict[str, Array]:
    if affine_terms_fn is None:
        raise ValueError("affine_terms_fn is required")
    f, g, y = affine_terms_fn(x, p)
    V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)

    LfV = gradV @ f
    LgV = gradV @ g
    LyV = gradV @ y

    b_scalar = -clf_cfg.lambda_clf * V - LfV - LyV * a_hat - jnp.abs(LyV) * radius

    return {
        "V": V,
        "gradV": gradV,
        "LfV": LfV,
        "LgV": LgV,
        "LyV": LyV,
        "b_scalar": b_scalar,
    }


def clf_shield(u_nom: Array,
               x: Array,
               adaptive_state: AdaptiveState,
               lyap_params: Dict[str, Any],
               lyap_cfg: LyapunovConfig,
               clf_cfg: CLFConfig,
               p: Any,
               affine_terms_fn: AffineTermsFn = None,
               alpha_max: float = 0.0,
               input_bounds: Optional[Tuple[Array, Array]] = None,
               ) -> Tuple[Array, Dict[str, Array]]:
    if affine_terms_fn is None:
        raise ValueError("affine_terms_fn is required")
    f, g, y = affine_terms_fn(x, p)
    V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
    LfV = gradV @ f
    LgV = gradV @ g
    LyV = gradV @ y

    if not clf_cfg.enabled:
        aux = {
            "V": V, "gradV": gradV,
            "LfV": LfV, "LgV": LgV, "LyV": LyV,
            "b_scalar": jnp.array(0.0),
            "violation": jnp.array(0.0),
            "final_violation": jnp.array(0.0),
            "projection_gain": jnp.array(0.0),
            "a_norm_sq": jnp.array(0.0),
            "feasible": jnp.array(1.0),
            "nominal_feasible": jnp.array(1.0),
            "box_feasible": jnp.array(1.0),
            "used_box_best_effort": jnp.array(0.0),
            "u_nom": u_nom, "u": u_nom,
        }
        return u_nom, aux

    b_scalar = (
        -clf_cfg.lambda_clf * V - LfV
        - LyV * adaptive_state.a_hat
        - jnp.abs(LyV) * adaptive_state.radius
    )

    if clf_cfg.enforce_input_bounds:
        bounds = input_bounds if input_bounds is not None else (p.u_min, p.u_max)
    else:
        bounds = None

    u, proj_aux = halfspace_projection(
        u_nom=jnp.atleast_1d(u_nom),
        a_vec=jnp.atleast_1d(LgV),
        b_scalar=b_scalar,
        eps=clf_cfg.eps_proj,
        clip_bounds=bounds,
        alpha_max=alpha_max,
    )

    aux = {
        "V": V, "gradV": gradV,
        "LfV": LfV, "LgV": LgV, "LyV": LyV,
        "b_scalar": b_scalar,
        "feasible": proj_aux["constraint_satisfied"],
    }
    aux.update(proj_aux)
    aux["u_nom"] = u_nom
    aux["u"] = u
    return u, aux
