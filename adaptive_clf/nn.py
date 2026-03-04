"""Small neural-network library and policy wrappers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from .configs import Array, MLPConfig


def _activation(name: str) -> Callable[[Array], Array]:
    name = name.lower()
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "gelu":
        return jax.nn.gelu
    if name == "softplus":
        return jax.nn.softplus
    if name in ("identity", "none", "linear"):
        return lambda x: x
    raise ValueError(f"Unknown activation: {name}")


def glorot_uniform(key: Array, shape: Tuple[int, ...]) -> Array:
    fan_in, fan_out = shape[-2], shape[-1]
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_mlp_params(key: Array, cfg: MLPConfig) -> Dict[str, Any]:
    sizes = (cfg.in_dim,) + tuple(cfg.hidden_sizes) + (cfg.out_dim,)
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for k, (din, dout) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = glorot_uniform(k, (din, dout))
        b = jnp.zeros((dout,))
        layers.append({"W": W, "b": b})
    return {"layers": tuple(layers)}


def apply_mlp(params: Dict[str, Any], x: Array, cfg: MLPConfig) -> Array:
    act = _activation(cfg.activation)
    final_act = _activation(cfg.final_activation)

    h = x
    for layer in params["layers"][:-1]:
        h = act(h @ layer["W"] + layer["b"])
    last = params["layers"][-1]
    y = h @ last["W"] + last["b"]
    y = final_act(y)
    return cfg.output_scale * y


def init_policy_params(key: Array, obs_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64)) -> Dict[str, Any]:
    cfg = MLPConfig(
        in_dim=obs_dim,
        hidden_sizes=hidden_sizes,
        out_dim=1,
        activation="tanh",
        final_activation="identity",
        output_scale=1.0,
    )
    return init_mlp_params(key, cfg)


def policy_apply(params: Dict[str, Any], obs: Array, u_min: float, u_max: float,
                 hidden_sizes: Tuple[int, ...] = (64, 64)) -> Array:
    cfg = MLPConfig(
        in_dim=obs.shape[-1],
        hidden_sizes=hidden_sizes,
        out_dim=1,
        activation="tanh",
        final_activation="identity",
        output_scale=1.0,
    )
    raw = apply_mlp(params, obs, cfg).squeeze(-1)
    return jnp.clip(u_max * jnp.tanh(raw), u_min, u_max)
