"""Generic RK4 integrator for control-affine systems."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from .configs import Array


def rk4_step_generic(
    dynamics_fn: Callable[[Array, Array, Array, Any], Array],
    x: Array,
    u: Array,
    a_true: Array,
    dt: float,
    p: Any,
    n_substeps: int = 1,
) -> Array:
    """Single RK4 step (or n_substeps sub-steps with the same control u).

    Parameters
    ----------
    dynamics_fn : callable with signature (x, u, a_true, p) -> xdot
    x : state vector
    u : control input (scalar or vector)
    a_true : true uncertainty parameter
    dt : timestep
    p : system parameters (any frozen dataclass)
    n_substeps : number of sub-steps within dt (for stiff systems)
    """
    sub_dt = dt / n_substeps

    def _one_rk4(x_i, _):
        k1 = dynamics_fn(x_i, u, a_true, p)
        k2 = dynamics_fn(x_i + 0.5 * sub_dt * k1, u, a_true, p)
        k3 = dynamics_fn(x_i + 0.5 * sub_dt * k2, u, a_true, p)
        k4 = dynamics_fn(x_i + sub_dt * k3, u, a_true, p)
        return x_i + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    if n_substeps == 1:
        k1 = dynamics_fn(x, u, a_true, p)
        k2 = dynamics_fn(x + 0.5 * dt * k1, u, a_true, p)
        k3 = dynamics_fn(x + 0.5 * dt * k2, u, a_true, p)
        k4 = dynamics_fn(x + dt * k3, u, a_true, p)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    x_final, _ = jax.lax.scan(_one_rk4, x, None, length=n_substeps)
    return x_final
