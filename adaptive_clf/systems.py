"""System registry: maps system names to dynamics, observations, costs, and defaults.

Each system provides:
  - params: frozen dataclass with physical parameters and control bounds
  - affine_terms_fn(x, p) -> (f, g, y): control-affine decomposition
  - dynamics_fn(x, u, a, p) -> xdot: full nonlinear dynamics
  - make_obs(x) -> obs: observation vector for the policy (may differ from state)
  - obs_dim: int
  - state_dim, ctrl_dim: int
  - x_eq: equilibrium state (tuple)
  - default_lqr_Q, default_lqr_R: LQR weight matrices
  - solve_lqr(p, Q, R, a_nom) -> (P, K)
  - default_cost_weights: dict of cost weights
  - sample_ics(key, batch_size, region_scale, a_true) -> (x0_batch, a_batch)
  - wrap_state(x) -> x: optional angle wrapping (identity if none needed)

Usage:
    from adaptive_clf.systems import get_system
    spec = get_system("cartpole")
    p = spec["params"]
    f, g, y = spec["affine_terms_fn"](x, p)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from .configs import (
    AcrobotParams, CartPoleParams, DubinsParams, PVTOLParams, Array,
)


# ---------------------------------------------------------------------------
# Acrobot
# ---------------------------------------------------------------------------

def _acrobot_spec() -> Dict[str, Any]:
    from .acrobot import (
        acrobot_affine_terms, acrobot_dynamics_true,
        solve_lqr_P,
    )

    def make_obs(x: Array) -> Array:
        return x  # raw state for acrobot

    def wrap_state(x: Array) -> Array:
        return x  # no wrapping needed (delta angles from upright)

    def sample_ics(key, batch_size, region_scale, a_true=0.0):
        low = jnp.array([-1.0, -1.0, -1.0, -1.0]) * region_scale
        high = jnp.array([1.0, 1.0, 1.0, 1.0]) * region_scale
        x0 = jax.random.uniform(key, (batch_size, 4), minval=low, maxval=high)
        a_batch = jnp.full((batch_size,), a_true)
        return x0, a_batch

    return {
        "name": "acrobot",
        "state_dim": 4,
        "ctrl_dim": 1,
        "obs_dim": 4,
        "params": AcrobotParams(),
        "affine_terms_fn": acrobot_affine_terms,
        "dynamics_fn": acrobot_dynamics_true,
        "make_obs": make_obs,
        "wrap_state": wrap_state,
        "x_eq": (0.0, 0.0, 0.0, 0.0),
        "default_lqr_Q": jnp.diag(jnp.array([40.0, 40.0, 8.0, 8.0])),
        "default_lqr_R": jnp.array([[0.5]]),
        "solve_lqr": solve_lqr_P,
        "sample_ics": sample_ics,
        "default_cost_weights": {
            "state_weights": (10.0, 10.0, 5.0, 5.0),
            "u_weight": 0.1,
            "terminal_scale": 10.0,
            "proj_weight": 0.1,
            "infeasible_weight": 50.0,
        },
    }


# ---------------------------------------------------------------------------
# Cart-pole
# ---------------------------------------------------------------------------

def _cartpole_spec() -> Dict[str, Any]:
    from .cartpole import (
        cartpole_affine_terms, cartpole_dynamics,
        solve_cartpole_lqr,
    )

    def make_obs(x: Array) -> Array:
        """[x_cart, sin(theta), cos(theta)-1, xdot, thetadot]"""
        return jnp.array([x[0], jnp.sin(x[1]), jnp.cos(x[1]) - 1.0, x[2], x[3]])

    def wrap_state(x: Array) -> Array:
        return x.at[1].set(jnp.arctan2(jnp.sin(x[1]), jnp.cos(x[1])))

    def sample_ics(key, batch_size, region_scale, a_true=0.0):
        k_bal, k_cur = jax.random.split(key)
        n_bal = max(1, batch_size // 2)
        n_cur = batch_size - n_bal

        bal_low = jnp.array([-0.5, -0.3, -0.5, -0.5]) * region_scale
        bal_high = jnp.array([0.5, 0.3, 0.5, 0.5]) * region_scale
        batch_bal = jax.random.uniform(k_bal, (n_bal, 4), minval=bal_low, maxval=bal_high)

        cur_low = jnp.array([-0.5, -jnp.pi, -0.5, -0.5]) * region_scale
        cur_high = jnp.array([0.5, jnp.pi, 0.5, 0.5]) * region_scale
        batch_cur = jax.random.uniform(k_cur, (n_cur, 4), minval=cur_low, maxval=cur_high)

        x0 = jnp.concatenate([batch_bal, batch_cur], axis=0)
        a_batch = jnp.full((batch_size,), a_true)
        return x0, a_batch

    return {
        "name": "cartpole",
        "state_dim": 4,
        "ctrl_dim": 1,
        "obs_dim": 5,
        "params": CartPoleParams(),
        "affine_terms_fn": cartpole_affine_terms,
        "dynamics_fn": cartpole_dynamics,
        "make_obs": make_obs,
        "wrap_state": wrap_state,
        "x_eq": (0.0, 0.0, 0.0, 0.0),
        "angle_indices": (1,),  # theta is an angle
        "default_lqr_Q": jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1])),
        "default_lqr_R": jnp.array([[0.01]]),
        "solve_lqr": solve_cartpole_lqr,
        "sample_ics": sample_ics,
        "default_cost_weights": {
            "state_weights": (1.0, 10.0, 0.5, 0.5),
            "u_weight": 0.01,
            "terminal_scale": 10.0,
            "proj_weight": 0.1,
            "infeasible_weight": 50.0,
        },
    }


# ---------------------------------------------------------------------------
# Dubins car
# ---------------------------------------------------------------------------

def _dubins_spec() -> Dict[str, Any]:
    from .dubins import (
        dubins_affine_terms, dubins_dynamics,
        solve_dubins_lqr,
    )

    def make_obs(x: Array) -> Array:
        """[e_x, e_y, sin(e_theta), cos(e_theta)-1]"""
        return jnp.array([x[0], x[1], jnp.sin(x[2]), jnp.cos(x[2]) - 1.0])

    def wrap_state(x: Array) -> Array:
        return x.at[2].set(jnp.arctan2(jnp.sin(x[2]), jnp.cos(x[2])))

    def sample_ics(key, batch_size, region_scale, a_true=0.0):
        low = jnp.array([-2.0, -2.0, -jnp.pi]) * region_scale
        high = jnp.array([2.0, 2.0, jnp.pi]) * region_scale
        x0 = jax.random.uniform(key, (batch_size, 3), minval=low, maxval=high)
        a_batch = jnp.full((batch_size,), a_true)
        return x0, a_batch

    p = DubinsParams()
    return {
        "name": "dubins",
        "state_dim": 3,
        "ctrl_dim": 2,
        "obs_dim": 4,
        "params": p,
        "affine_terms_fn": dubins_affine_terms,
        "dynamics_fn": dubins_dynamics,
        "make_obs": make_obs,
        "wrap_state": wrap_state,
        "x_eq": (0.0, 0.0, 0.0),
        "angle_indices": (2,),  # e_theta is an angle
        "u_eq": jnp.array([p.v_ref, 0.0]),
        "u_min": jnp.array([p.v_min, p.omega_min]),
        "u_max": jnp.array([p.v_max, p.omega_max]),
        "default_lqr_Q": jnp.diag(jnp.array([0.5, 5.0, 2.0])),
        "default_lqr_R": jnp.diag(jnp.array([0.1, 0.1])),
        "solve_lqr": solve_dubins_lqr,
        "sample_ics": sample_ics,
        "default_cost_weights": {
            "state_weights": (0.5, 5.0, 2.0),
            "u_weight": 0.01,
            "terminal_scale": 10.0,
            "proj_weight": 0.1,
            "infeasible_weight": 50.0,
        },
    }


# ---------------------------------------------------------------------------
# PVTOL
# ---------------------------------------------------------------------------

def _pvtol_spec() -> Dict[str, Any]:
    from .pvtol import (
        pvtol_affine_terms, pvtol_dynamics,
        solve_pvtol_lqr,
    )

    p = PVTOLParams()

    def make_obs(x: Array) -> Array:
        """[px, py, sin(theta), cos(theta)-1, vx, vy, thetadot]"""
        return jnp.array([
            x[0], x[1],
            jnp.sin(x[2]), jnp.cos(x[2]) - 1.0,
            x[3], x[4], x[5],
        ])

    def wrap_state(x: Array) -> Array:
        return x.at[2].set(jnp.arctan2(jnp.sin(x[2]), jnp.cos(x[2])))

    def sample_ics(key, batch_size, region_scale, a_true=0.0):
        low = jnp.array([-2.0, -2.0, -0.5, -1.0, -1.0, -1.0]) * region_scale
        high = jnp.array([2.0, 2.0, 0.5, 1.0, 1.0, 1.0]) * region_scale
        x0 = jax.random.uniform(key, (batch_size, 6), minval=low, maxval=high)
        a_batch = jnp.full((batch_size,), a_true)
        return x0, a_batch

    return {
        "name": "pvtol",
        "state_dim": 6,
        "ctrl_dim": 2,
        "obs_dim": 7,
        "params": p,
        "affine_terms_fn": pvtol_affine_terms,
        "dynamics_fn": pvtol_dynamics,
        "make_obs": make_obs,
        "wrap_state": wrap_state,
        "x_eq": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "angle_indices": (2,),  # theta is an angle
        "u_eq": jnp.array([p.m * p.g, 0.0]),
        "u_min": jnp.array([p.T_min, p.tau_min]),
        "u_max": jnp.array([p.T_max, p.tau_max]),
        "default_lqr_Q": jnp.diag(jnp.array([2.0, 2.0, 5.0, 0.5, 0.5, 0.5])),
        "default_lqr_R": jnp.diag(jnp.array([0.01, 0.1])),
        "solve_lqr": solve_pvtol_lqr,
        "sample_ics": sample_ics,
        "default_cost_weights": {
            "state_weights": (2.0, 2.0, 5.0, 0.5, 0.5, 0.5),
            "u_weight": 0.01,
            "terminal_scale": 10.0,
            "proj_weight": 0.1,
            "infeasible_weight": 50.0,
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[], Dict[str, Any]]] = {
    "acrobot": _acrobot_spec,
    "cartpole": _cartpole_spec,
    "dubins": _dubins_spec,
    "pvtol": _pvtol_spec,
}


def get_system(name: str) -> Dict[str, Any]:
    """Get system spec by name. Raises KeyError if not found."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown system '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def list_systems() -> list:
    return list(_REGISTRY.keys())
