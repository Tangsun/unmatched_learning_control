"""Configuration dataclasses and shared type aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Tuple

import jax.numpy as jnp

Array = jnp.ndarray
PyTree = Any


@dataclass(frozen=True)
class AcrobotParams:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    lc1: float = 0.5
    lc2: float = 0.5
    I1: float = 1 / 3
    I2: float = 1 / 3
    g: float = 9.81

    u_min: float = -100.0
    u_max: float = 100.0

    a_min: float = 0.0
    a_max: float = 2.0

    eps: float = 1e-8


@dataclass(frozen=True)
class MLPConfig:
    in_dim: int
    hidden_sizes: Tuple[int, ...] = (64, 64)
    out_dim: int = 1
    activation: str = "tanh"
    final_activation: str = "identity"
    output_scale: float = 1.0


@dataclass(frozen=True)
class LyapunovConfig:
    mode: str = "quadratic_fixed"
    state_dim: int = 4
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"

    x_eq: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    P_init: Optional[Array] = None

    eps_pd: float = 1e-3


@dataclass(frozen=True)
class AdaptiveConfig:
    eta: float = 2e-2
    info_init: float = 1e-3
    radius_floor: float = 1e-3
    radius_scale: float = 1.0
    stopgrad_obs: bool = True
    adapt_enabled: bool = False


@dataclass(frozen=True)
class CLFConfig:
    lambda_clf: float = 0.25
    eps_proj: float = 1e-8
    enforce_input_bounds: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class RolloutConfig:
    horizon: int = 200
    dt: float = 0.02

    Q_track: Tuple[Tuple[float, ...], ...] = (
        (10.0, 0.0, 0.0, 0.0),
        (0.0, 10.0, 0.0, 0.0),
        (0.0, 0.0, 5.0, 0.0),
        (0.0, 0.0, 0.0, 5.0),
    )
    Q_terminal_scale: float = 0.0

    R_u: float = 1e-1
    R_proj: float = 1e-1
    R_infeasible: float = 50.0

    R_energy: float = 0.0

    x_max: float = 10.0
    u_clip: float = 100.0

    n_substeps: int = 1
    bptt_window: int = 0

    lqr_blend: bool = False
    lqr_V_threshold: float = 5.0
    lqr_temperature: float = 1.0


@dataclass(frozen=True)
class CartPoleParams:
    mc: float = 1.0
    mp: float = 0.1
    l: float = 0.5
    g: float = 9.81

    u_min: float = -10.0
    u_max: float = 10.0

    a_min: float = 0.0
    a_max: float = 2.0

    eps: float = 1e-8


@dataclass(frozen=True)
class DubinsParams:
    v_ref: float = 1.0

    v_min: float = 0.0
    v_max: float = 2.0
    omega_min: float = -2.0
    omega_max: float = 2.0

    a_min: float = -0.5
    a_max: float = 0.5


class AdaptiveState(NamedTuple):
    a_hat: Array
    info: Array
    radius: Array
