"""Lyapunov function modules (fixed quadratic, learned quadratic, MLP-PSD)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from .configs import Array, LyapunovConfig, MLPConfig
from .nn import apply_mlp, init_mlp_params


def _lower_tri_from_vector(v: Array, n: int) -> Array:
    L = jnp.zeros((n, n), dtype=v.dtype)
    idx = jnp.tril_indices(n, k=-1)
    return L.at[idx].set(v)


def _spd_matrix_from_params(diag_log: Array, off_diag: Array) -> Array:
    n = diag_log.shape[0]
    L = _lower_tri_from_vector(off_diag, n)
    L = L + jnp.diag(jnp.exp(diag_log))
    return L @ L.T


def _pendulum_energy(x: Array, mp: float, l: float, g: float) -> Array:
    """Pendulum mechanical energy relative to upright (E=0 at upright).

    E = 0.5 * mp * l^2 * thetadot^2 + mp * g * l * (cos(theta) - 1)

    State: x = [x_cart, theta, x_cart_dot, theta_dot].
    """
    theta, thetadot = x[1], x[3]
    KE = 0.5 * mp * l ** 2 * thetadot ** 2
    PE = mp * g * l * (jnp.cos(theta) - 1.0)
    return KE + PE


def init_lyapunov_params(key: Array, cfg: LyapunovConfig) -> Dict[str, Any]:
    x_eq = jnp.asarray(cfg.x_eq, dtype=jnp.float32)

    if cfg.mode == "quadratic_fixed":
        if cfg.P_init is None:
            raise ValueError("quadratic_fixed requires cfg.P_init.")
        P = jnp.asarray(cfg.P_init, dtype=jnp.float32)
        return {"mode": cfg.mode, "x_eq": x_eq, "P": P}

    if cfg.mode == "quadratic_learned":
        n = cfg.state_dim
        if cfg.P_init is None:
            P0 = jnp.eye(n, dtype=jnp.float32)
        else:
            P0 = jnp.asarray(cfg.P_init, dtype=jnp.float32)
        L0 = jnp.linalg.cholesky(P0 + 1e-6 * jnp.eye(n))
        diag_log = jnp.log(jnp.diag(L0))
        off_diag = L0[jnp.tril_indices(n, k=-1)]
        return {"mode": cfg.mode, "x_eq": x_eq, "diag_log": diag_log, "off_diag": off_diag}

    if cfg.mode == "mlp_psd":
        mlp_cfg = MLPConfig(
            in_dim=cfg.state_dim,
            hidden_sizes=cfg.hidden_sizes,
            out_dim=1,
            activation=cfg.activation,
            final_activation="identity",
            output_scale=1.0,
        )
        phi_params = init_mlp_params(key, mlp_cfg)
        return {"mode": cfg.mode, "x_eq": x_eq, "phi": phi_params}

    if cfg.mode == "energy_cartpole":
        if cfg.energy_phys is None:
            raise ValueError("energy_cartpole requires cfg.energy_phys = (mp, l, g).")
        mp, l, g = cfg.energy_phys
        w_cart, w_vel = cfg.energy_weights
        return {
            "mode": cfg.mode,
            "x_eq": x_eq,
            "mp": jnp.float32(mp),
            "l": jnp.float32(l),
            "g": jnp.float32(g),
            "w_cart": jnp.float32(w_cart),
            "w_vel": jnp.float32(w_vel),
        }

    raise ValueError(f"Unknown Lyapunov mode: {cfg.mode}")


def lyapunov_matrix(lyap_params: Dict[str, Any], lyap_cfg: LyapunovConfig) -> Optional[Array]:
    mode = lyap_params["mode"]
    if mode == "quadratic_fixed":
        return lyap_params["P"]
    if mode == "quadratic_learned":
        return _spd_matrix_from_params(lyap_params["diag_log"], lyap_params["off_diag"])
    return None


def _wrap_angle_diff(diff: Array, angle_indices: Tuple[int, ...]) -> Array:
    """Wrap angular components of diff to [-pi, pi]."""
    for i in angle_indices:
        diff = diff.at[i].set(jnp.arctan2(jnp.sin(diff[i]), jnp.cos(diff[i])))
    return diff


def lyapunov_value(lyap_params: Dict[str, Any], lyap_cfg: LyapunovConfig, x: Array) -> Array:
    mode = lyap_params["mode"]
    x_eq = lyap_params["x_eq"]
    diff = x - x_eq

    # Wrap angular differences so theta=pi gives diff=pi (not 0 after mod 2pi)
    if lyap_cfg.angle_indices:
        diff = _wrap_angle_diff(diff, lyap_cfg.angle_indices)

    if mode in ("quadratic_fixed", "quadratic_learned"):
        P = lyapunov_matrix(lyap_params, lyap_cfg)
        return diff @ P @ diff

    if mode == "mlp_psd":
        mlp_cfg = MLPConfig(
            in_dim=lyap_cfg.state_dim,
            hidden_sizes=lyap_cfg.hidden_sizes,
            out_dim=1,
            activation=lyap_cfg.activation,
            final_activation="identity",
            output_scale=1.0,
        )
        zero = jnp.zeros_like(diff)
        phi_x = apply_mlp(lyap_params["phi"], diff, mlp_cfg).squeeze(-1)
        phi_0 = apply_mlp(lyap_params["phi"], zero, mlp_cfg).squeeze(-1)
        delta = phi_x - phi_0
        return 0.5 * delta ** 2 + lyap_cfg.eps_pd * (diff @ diff)

    if mode == "energy_cartpole":
        mp = lyap_params["mp"]
        l = lyap_params["l"]
        g = lyap_params["g"]
        w_cart = lyap_params["w_cart"]
        w_vel = lyap_params["w_vel"]
        E = _pendulum_energy(x, mp, l, g)
        x_cart, x_cart_dot = x[0], x[2]
        return E ** 2 + w_cart * x_cart ** 2 + w_vel * x_cart_dot ** 2

    raise ValueError(f"Unknown Lyapunov mode: {mode}")


def lyapunov_value_and_grad(lyap_params: Dict[str, Any],
                            lyap_cfg: LyapunovConfig,
                            x: Array) -> Tuple[Array, Array]:
    mode = lyap_params["mode"]
    x_eq = lyap_params["x_eq"]
    diff = x - x_eq

    # Use analytic gradient only when no angle wrapping is needed
    if mode in ("quadratic_fixed", "quadratic_learned") and not lyap_cfg.angle_indices:
        P = lyapunov_matrix(lyap_params, lyap_cfg)
        V = diff @ P @ diff
        gradV = 2.0 * (P @ diff)
        return V, gradV

    # Generic path: jax.grad handles wrapping chain rule automatically
    V_fun = lambda z: lyapunov_value(lyap_params, lyap_cfg, z)
    V = V_fun(x)
    gradV = jax.grad(V_fun)(x)
    return V, gradV
