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
    """Euclidean projection onto {u : a^T u <= b}, optionally clipped to a box.

    Parameters
    ----------
    u_nom : nominal control input (any dimension).
    a_vec : constraint normal (same shape as u_nom).
    b_scalar : constraint right-hand side.
    eps : regularization for near-zero norm of a_vec.
    clip_bounds : if provided, (u_min, u_max) box applied after projection.
        For scalar u this is exact (interval intersection).  For multi-input
        u it is a heuristic post-hoc clip that can violate the half-space
        constraint -- a proper joint solver should be used instead.
    alpha_max : if > 0, cap the projection gain to prevent gradient explosion
        when ||a_vec||^2 is small (infeasible states).
    """
    a_vec = jnp.atleast_1d(a_vec)
    u_nom = jnp.atleast_1d(u_nom)
    violation = jnp.dot(a_vec, u_nom) - b_scalar
    denom = jnp.maximum(jnp.dot(a_vec, a_vec), eps)
    alpha = jax.nn.relu(violation) / denom
    if alpha_max > 0:
        alpha = jnp.minimum(alpha, alpha_max)
    u_proj = u_nom - alpha * a_vec

    if clip_bounds is not None:
        u_proj = jnp.clip(u_proj, clip_bounds[0], clip_bounds[1])

    aux = {
        "violation": violation,
        "projection_gain": alpha,
        "a_norm_sq": jnp.dot(a_vec, a_vec),
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
            "projection_gain": jnp.array(0.0),
            "a_norm_sq": jnp.array(0.0),
            "feasible": jnp.array(1.0),
            "u_nom": u_nom, "u": u_nom,
        }
        return u_nom, aux

    b_scalar = (
        -clf_cfg.lambda_clf * V - LfV
        - LyV * adaptive_state.a_hat
        - jnp.abs(LyV) * adaptive_state.radius
    )

    bounds = (p.u_min, p.u_max) if clf_cfg.enforce_input_bounds else None

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
        "feasible": jnp.where(proj_aux["violation"] <= 0, 1.0, 0.0),
    }
    aux.update(proj_aux)
    aux["u_nom"] = u_nom
    aux["u"] = u
    return u, aux
