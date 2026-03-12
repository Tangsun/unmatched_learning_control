#!/usr/bin/env python3
"""Visualize learned Lyapunov function for Dubins car.

Creates 2D slice plots of V(x), ||LgV||, |LyV|, and the robust CLF margin
to understand where the shield is effective and where it struggles.

Usage:
    python scripts/visualize_lyapunov_dubins.py \
        --run-dir runs/dubins_observer_p2a_fast
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adaptive_clf.configs import DubinsParams, LyapunovConfig, CLFConfig
from adaptive_clf.dubins import dubins_affine_terms
from adaptive_clf.lyapunov import lyapunov_value, lyapunov_value_and_grad


def load_run(run_dir: str):
    with open(os.path.join(run_dir, "policy_params.pkl"), "rb") as f:
        data = pickle.load(f)
    lyap_params = data["lyap_params"]
    lyap_cfg = data["lyap_cfg"]
    return data, lyap_params, lyap_cfg


def compute_grid(lyap_params, lyap_cfg, p, grid_x, grid_y, fixed_dim, fixed_val):
    """Compute V, ||LgV||^2, |LyV|, and CLF margin on a 2D grid.

    fixed_dim: which of (e_x=0, e_y=1, e_theta=2) is held fixed
    The other two dims are swept over grid_x and grid_y.
    """
    nx, ny = len(grid_x), len(grid_y)
    xx, yy = jnp.meshgrid(grid_x, grid_y, indexing="ij")
    points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # (N, 2)

    # Build full 3D state
    def make_state(pt):
        if fixed_dim == 0:
            return jnp.array([fixed_val, pt[0], pt[1]])
        elif fixed_dim == 1:
            return jnp.array([pt[0], fixed_val, pt[1]])
        else:
            return jnp.array([pt[0], pt[1], fixed_val])

    def compute_one(pt):
        x = make_state(pt)
        V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
        f, g, y = dubins_affine_terms(x, p)
        LfV = gradV @ f
        LgV = gradV @ g          # (2,)
        LyV = gradV @ y          # scalar
        LgV_norm_sq = jnp.dot(LgV, LgV)
        return V, LgV_norm_sq, jnp.abs(LyV), LfV, LgV

    results = jax.vmap(compute_one)(points)
    V = results[0].reshape(nx, ny)
    LgV_sq = results[1].reshape(nx, ny)
    absLyV = results[2].reshape(nx, ny)
    LfV = results[3].reshape(nx, ny)
    LgV_vec = results[4].reshape(nx, ny, 2)
    return xx, yy, V, LgV_sq, absLyV, LfV, LgV_vec


def plot_slices(lyap_params, lyap_cfg, p, lambda_clf, out_dir, run_name):
    """Create three rows of slice plots: e_x=0, e_y=0, e_theta=0."""
    os.makedirs(out_dir, exist_ok=True)

    slices = [
        {"fixed_dim": 0, "fixed_val": 0.0, "xlabel": "$e_y$", "ylabel": "$e_\\theta$",
         "title_suffix": "$e_x = 0$", "range_x": (-2, 2), "range_y": (-np.pi, np.pi)},
        {"fixed_dim": 1, "fixed_val": 0.0, "xlabel": "$e_x$", "ylabel": "$e_\\theta$",
         "title_suffix": "$e_y = 0$", "range_x": (-2, 2), "range_y": (-np.pi, np.pi)},
        {"fixed_dim": 2, "fixed_val": 0.0, "xlabel": "$e_x$", "ylabel": "$e_y$",
         "title_suffix": "$e_\\theta = 0$", "range_x": (-2, 2), "range_y": (-2, 2)},
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Lyapunov Analysis: {run_name}", fontsize=14)

    for row, sl in enumerate(slices):
        grid_x = jnp.linspace(sl["range_x"][0], sl["range_x"][1], 100)
        grid_y = jnp.linspace(sl["range_y"][0], sl["range_y"][1], 100)
        xx, yy, V, LgV_sq, absLyV, LfV, LgV_vec = compute_grid(
            lyap_params, lyap_cfg, p, grid_x, grid_y,
            sl["fixed_dim"], sl["fixed_val"])

        V_np = np.array(V)
        LgV_sq_np = np.array(LgV_sq)
        absLyV_np = np.array(absLyV)
        LfV_np = np.array(LfV)
        xx_np = np.array(xx)
        yy_np = np.array(yy)

        # Panel 1: V(x) level sets
        ax = axes[row, 0]
        levels = np.linspace(0, min(V_np.max(), 5.0), 25)
        cf = ax.contourf(xx_np, yy_np, V_np, levels=levels, cmap="viridis")
        ax.contour(xx_np, yy_np, V_np, levels=[0.5, 1.0, 2.0, 3.0], colors="white",
                   linewidths=0.8, linestyles="--")
        plt.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_title(f"V(x) | {sl['title_suffix']}")
        ax.set_xlabel(sl["xlabel"])
        ax.set_ylabel(sl["ylabel"])
        ax.plot(0, 0, "r*", markersize=10)

        # Panel 2: ||LgV||^2 — where the shield has control authority
        ax = axes[row, 1]
        eps_proj = 0.1
        cf = ax.contourf(xx_np, yy_np, np.log10(np.maximum(LgV_sq_np, 1e-10)),
                         levels=np.linspace(-4, 2, 25), cmap="RdYlGn")
        ax.contour(xx_np, yy_np, LgV_sq_np, levels=[eps_proj], colors="red",
                   linewidths=2, linestyles="-")
        plt.colorbar(cf, ax=ax, shrink=0.8, label="log10")
        ax.set_title(f"$||L_g V||^2$ | {sl['title_suffix']}\n(red = eps_proj={eps_proj})")
        ax.set_xlabel(sl["xlabel"])
        ax.set_ylabel(sl["ylabel"])
        ax.plot(0, 0, "r*", markersize=10)

        # Panel 3: |LyV| — sensitivity to uncertainty
        ax = axes[row, 2]
        cf = ax.contourf(xx_np, yy_np, absLyV_np, levels=25, cmap="magma")
        ax.contour(xx_np, yy_np, absLyV_np, levels=[0.1, 0.5, 1.0], colors="white",
                   linewidths=0.8, linestyles="--")
        plt.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_title(f"|$L_y V$| | {sl['title_suffix']}\n(uncertainty sensitivity)")
        ax.set_xlabel(sl["xlabel"])
        ax.set_ylabel(sl["ylabel"])
        ax.plot(0, 0, "r*", markersize=10)

        # Panel 4: Robust CLF "slack" = -LfV - lambda*V (before control)
        # Positive means the drift alone satisfies the constraint;
        # negative means the shield must compensate
        slack = -LfV_np - lambda_clf * V_np
        ax = axes[row, 3]
        vmax = max(abs(slack.min()), abs(slack.max()), 1.0)
        cf = ax.contourf(xx_np, yy_np, slack,
                         levels=np.linspace(-vmax, vmax, 25), cmap="RdBu")
        ax.contour(xx_np, yy_np, slack, levels=[0], colors="black",
                   linewidths=2, linestyles="-")
        plt.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_title(f"$-L_f V - \\lambda V$ | {sl['title_suffix']}\n(>0: drift OK, <0: shield needed)")
        ax.set_xlabel(sl["xlabel"])
        ax.set_ylabel(sl["ylabel"])
        ax.plot(0, 0, "r*", markersize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "lyapunov_slices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_gradient_field(lyap_params, lyap_cfg, p, out_dir, run_name):
    """Plot gradV direction field overlaid on V contours for the e_x-e_y plane."""
    os.makedirs(out_dir, exist_ok=True)

    grid = jnp.linspace(-2, 2, 60)
    xx, yy = jnp.meshgrid(grid, grid, indexing="ij")
    points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    def compute_one(pt):
        x = jnp.array([pt[0], pt[1], 0.0])  # e_theta=0
        V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
        return V, gradV[:2]  # just e_x, e_y components

    results = jax.vmap(compute_one)(points)
    V = np.array(results[0].reshape(60, 60))
    gx = np.array(results[1][:, 0].reshape(60, 60))
    gy = np.array(results[1][:, 1].reshape(60, 60))
    xx_np = np.array(xx)
    yy_np = np.array(yy)

    # Normalize for quiver
    mag = np.sqrt(gx**2 + gy**2) + 1e-8
    gx_n, gy_n = gx / mag, gy / mag

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    levels = np.linspace(0, min(V.max(), 5.0), 30)
    cf = ax.contourf(xx_np, yy_np, V, levels=levels, cmap="viridis", alpha=0.8)
    ax.contour(xx_np, yy_np, V, levels=[0.5, 1.0, 2.0, 3.0], colors="white",
               linewidths=1, linestyles="--")
    plt.colorbar(cf, ax=ax, label="V(x)")

    # Quiver every 4th point
    skip = 4
    ax.quiver(xx_np[::skip, ::skip], yy_np[::skip, ::skip],
              gx_n[::skip, ::skip], gy_n[::skip, ::skip],
              color="red", alpha=0.6, scale=25, width=0.004)

    ax.set_xlabel("$e_x$")
    ax.set_ylabel("$e_y$")
    ax.set_title(f"V(x) contours + $\\nabla V$ direction | $e_\\theta=0$ | {run_name}")
    ax.plot(0, 0, "r*", markersize=15)
    ax.set_aspect("equal")

    path = os.path.join(out_dir, "lyapunov_gradient_field.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_1d_cuts(lyap_params, lyap_cfg, p, lambda_clf, out_dir, run_name):
    """1D cuts through the origin along each axis."""
    os.makedirs(out_dir, exist_ok=True)

    t = jnp.linspace(-2, 2, 200)

    directions = [
        ("$e_x$", lambda s: jnp.array([s, 0.0, 0.0])),
        ("$e_y$", lambda s: jnp.array([0.0, s, 0.0])),
        ("$e_\\theta$", lambda s: jnp.array([0.0, 0.0, s])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"1D Cuts Through Origin | {run_name}", fontsize=14)

    for col, (label, make_x) in enumerate(directions):
        def compute_one(s, make_x=make_x):
            x = make_x(s)
            V, gradV = lyapunov_value_and_grad(lyap_params, lyap_cfg, x)
            f, g, y = dubins_affine_terms(x, p)
            LfV = gradV @ f
            LgV = gradV @ g
            LyV = gradV @ y
            return V, LfV, jnp.dot(LgV, LgV), jnp.abs(LyV)

        results = jax.vmap(compute_one)(t)
        V_arr = np.array(results[0])
        LfV_arr = np.array(results[1])
        LgV_sq_arr = np.array(results[2])
        absLyV_arr = np.array(results[3])
        t_np = np.array(t)

        # Top row: V(x)
        ax = axes[0, col]
        ax.plot(t_np, V_arr, "b-", linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel("V(x)")
        ax.set_title(f"V along {label}")
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Bottom row: ||LgV||^2 and |LyV|
        ax = axes[1, col]
        ax.plot(t_np, LgV_sq_arr, "g-", linewidth=2, label="$||L_g V||^2$")
        ax.plot(t_np, absLyV_arr, "r-", linewidth=2, label="$|L_y V|$")
        ax.axhline(0.1, color="gray", linestyle=":", alpha=0.7, label="$\\epsilon_{proj}$")
        ax.set_xlabel(label)
        ax.set_ylabel("magnitude")
        ax.set_title(f"Shield terms along {label}")
        ax.legend(fontsize=8)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "lyapunov_1d_cuts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lambda-clf", type=float, default=0.2)
    args = parser.parse_args()

    data, lyap_params, lyap_cfg = load_run(args.run_dir)
    p = DubinsParams()
    run_name = os.path.basename(args.run_dir)
    out_dir = os.path.join(args.run_dir, "lyapunov_viz")

    print(f"Visualizing Lyapunov function from: {args.run_dir}")
    print(f"  mode: {lyap_params['mode']}")
    print(f"  lyap_cfg: {lyap_cfg}")

    plot_slices(lyap_params, lyap_cfg, p, args.lambda_clf, out_dir, run_name)
    plot_gradient_field(lyap_params, lyap_cfg, p, out_dir, run_name)
    plot_1d_cuts(lyap_params, lyap_cfg, p, args.lambda_clf, out_dir, run_name)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
