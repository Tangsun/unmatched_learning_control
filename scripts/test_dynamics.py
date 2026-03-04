"""Sanity-check the Acrobot dynamics: zero-torque free response and LQR stabilization.

Run from the repo root:
    python scripts/test_dynamics.py

Produces scripts/dynamics_test.png with six panels:
  Top row    – zero-torque simulation for several friction values (a = 0, 1, 2)
  Bottom row – LQR feedback for nominal (a=0) and with mild friction (a=0.5)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from adaptive_clf.configs import AcrobotParams
from adaptive_clf.acrobot import (
    acrobot_dynamics_true,
    acrobot_terms,
    rk4_step,
    solve_lqr_P,
)


def compute_energy(x: jnp.ndarray, p: AcrobotParams) -> float:
    """Total mechanical energy: kinetic + gravitational potential.

    Potential energy reference: both links hanging straight down (q1=0, q2=0).
    """
    dq1, dq2, w1, w2 = x
    q1 = jnp.pi + dq1
    q2 = dq2
    qdot = jnp.array([w1, w2])

    M, _, _ = acrobot_terms(x, p)
    KE = 0.5 * qdot @ M @ qdot

    y1 = -p.lc1 * jnp.cos(q1)
    y2 = -p.l1 * jnp.cos(q1) - p.lc2 * jnp.cos(q1 + q2)
    PE = p.m1 * p.g * y1 + p.m2 * p.g * y2
    return KE + PE


def simulate(x0, u_fn, a_true, p, dt, n_steps):
    """Plain loop simulation returning (n_steps+1, 4) trajectory and energy."""
    xs = [x0]
    energies = [float(compute_energy(x0, p))]
    us = []
    x = x0
    for _ in range(n_steps):
        u = u_fn(x)
        us.append(float(u))
        x = rk4_step(acrobot_dynamics_true, x, u, a_true, dt, p)
        xs.append(x)
        energies.append(float(compute_energy(x, p)))
    return np.array(xs), np.array(energies), np.array(us)


def main():
    p = AcrobotParams()
    dt = 0.02
    n_steps = 250
    t = np.arange(n_steps + 1) * dt

    x0_free = jnp.array([0.1, 0.0, 0.0, 0.0], dtype=jnp.float32)
    x0_lqr = jnp.array([0.01, 0.0, 0.0, 0.0], dtype=jnp.float32)

    # --- Zero-torque simulations ---
    zero_u = lambda x: jnp.array(0.0)
    friction_values = [0.0, 1.0, 2.0]
    zt_results = {}
    for a in friction_values:
        xs, es, us = simulate(x0_free, zero_u, jnp.array(a), p, dt, n_steps)
        zt_results[a] = (xs, es, us)

    # --- LQR simulations (designed for a_nom=0) ---
    Q_lqr = jnp.diag(jnp.array([40.0, 40.0, 8.0, 8.0]))
    R_lqr = jnp.array([[0.5]])
    _, K = solve_lqr_P(p, Q=Q_lqr, R=R_lqr, a_nom=0.0)
    K_np = np.array(K).squeeze()

    lqr_u = lambda x: jnp.clip(-K_np @ x, p.u_min, p.u_max)

    lqr_cases = {"a=0 (nominal)": 0.0, "a=0.5": 0.5}
    lqr_results = {}
    for label, a in lqr_cases.items():
        xs, es, us = simulate(x0_lqr, lqr_u, jnp.array(a), p, dt, n_steps)
        lqr_results[label] = (xs, es, us)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Top row: zero-torque
    labels = {0.0: "a=0 (no friction)", 1.0: "a=1", 2.0: "a=2"}
    for a, (xs, es, us) in zt_results.items():
        lbl = labels[a]
        axes[0, 0].plot(t, xs[:, 0], label=f"$\\delta q_1$ {lbl}")
        axes[0, 0].plot(t, xs[:, 1], "--", label=f"$\\delta q_2$ {lbl}")
        axes[0, 1].plot(t, xs[:, 2], label=f"$\\dot q_1$ {lbl}")
        axes[0, 1].plot(t, xs[:, 3], "--", label=f"$\\dot q_2$ {lbl}")
        axes[0, 2].plot(t, es, label=lbl)

    axes[0, 0].set_title("Joint angles (u=0)")
    axes[0, 0].set_ylabel("rad")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Joint velocities (u=0)")
    axes[0, 1].set_ylabel("rad/s")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_title("Total energy (u=0)")
    axes[0, 2].set_ylabel("energy")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].set_xlabel("time (s)")
    axes[0, 2].grid(True, alpha=0.3)

    # Bottom row: LQR
    t_u = t[:-1]
    for label, (xs, es, us) in lqr_results.items():
        axes[1, 0].plot(t, xs[:, 0], label=f"$\\delta q_1$ {label}")
        axes[1, 0].plot(t, xs[:, 1], "--", label=f"$\\delta q_2$ {label}")
        axes[1, 1].plot(t, xs[:, 2], label=f"$\\dot q_1$ {label}")
        axes[1, 1].plot(t, xs[:, 3], "--", label=f"$\\dot q_2$ {label}")
        axes[1, 2].plot(t_u, us, label=label)

    axes[1, 0].set_title(f"Joint angles (LQR, x0={float(x0_lqr[0])})")
    axes[1, 0].set_ylabel("rad")
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Joint velocities (LQR)")
    axes[1, 1].set_ylabel("rad/s")
    axes[1, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_title("Control input (LQR)")
    axes[1, 2].set_ylabel("torque")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_xlabel("time (s)")
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Acrobot dynamics test (I about pivots: I1={p.I1:.3f}, I2={p.I2:.3f})",
        fontsize=13,
    )
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "dynamics_test.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    # Print summary
    print("\n--- Zero-torque (x0=[0.1, 0, 0, 0]) ---")
    for a, (xs, es, _) in zt_results.items():
        e_drift = abs(es[-1] - es[0])
        e_rel = e_drift / (abs(es[0]) + 1e-12)
        print(f"  a={a:.0f}: energy drift = {e_drift:.6f} ({e_rel:.2%}), final |x| = {np.linalg.norm(xs[-1]):.4f}")

    print(f"\n--- LQR (x0=[{float(x0_lqr[0])}, 0, 0, 0], designed for a_nom=0) ---")
    for label, (xs, es, _) in lqr_results.items():
        print(f"  {label}: final |x| = {np.linalg.norm(xs[-1]):.6f}")
    print("  Note: LQR fails at a=1.0 (too large a model mismatch) -- motivates adaptive control!")


if __name__ == "__main__":
    main()
