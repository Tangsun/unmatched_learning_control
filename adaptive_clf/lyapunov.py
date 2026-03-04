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

    raise ValueError(f"Unknown Lyapunov mode: {cfg.mode}")


def lyapunov_matrix(lyap_params: Dict[str, Any], lyap_cfg: LyapunovConfig) -> Optional[Array]:
    mode = lyap_params["mode"]
    if mode == "quadratic_fixed":
        return lyap_params["P"]
    if mode == "quadratic_learned":
        return _spd_matrix_from_params(lyap_params["diag_log"], lyap_params["off_diag"])
    return None


def lyapunov_value(lyap_params: Dict[str, Any], lyap_cfg: LyapunovConfig, x: Array) -> Array:
    mode = lyap_params["mode"]
    x_eq = lyap_params["x_eq"]
    diff = x - x_eq

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

    raise ValueError(f"Unknown Lyapunov mode: {mode}")


def lyapunov_value_and_grad(lyap_params: Dict[str, Any],
                            lyap_cfg: LyapunovConfig,
                            x: Array) -> Tuple[Array, Array]:
    mode = lyap_params["mode"]
    x_eq = lyap_params["x_eq"]
    diff = x - x_eq

    if mode in ("quadratic_fixed", "quadratic_learned"):
        P = lyapunov_matrix(lyap_params, lyap_cfg)
        V = diff @ P @ diff
        gradV = 2.0 * (P @ diff)
        return V, gradV

    V_fun = lambda z: lyapunov_value(lyap_params, lyap_cfg, z)
    V = V_fun(x)
    gradV = jax.grad(V_fun)(x)
    return V, gradV
