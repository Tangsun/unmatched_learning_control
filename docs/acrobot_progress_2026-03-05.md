# Acrobot Stabilization via Differentiable Simulation -- Progress Report

**Date:** 2026-03-05  
**Status:** Paused -- switching focus to cart-pole

## Goal

Train a neural network control policy to stabilize the acrobot (underactuated double pendulum) at the upright equilibrium, starting from arbitrary initial conditions (swing-up + stabilization). Three candidate approaches were explored:

1. **LQR blending** -- use a linear-quadratic regulator near upright, NN elsewhere  
2. **Control Lyapunov Function (CLF)** -- enforce a Lyapunov decrease constraint via projection  
3. **Curriculum training** -- gradually expand the training region from near-upright outward

## System Description

- **State:** `x = [dq1, dq2, q1dot, q2dot]` where `q1 = pi + dq1`, `q2 = dq2` (upright = origin)  
- **Control:** Torque at elbow joint only (underactuated -- no shoulder torque)  
- **Dynamics:** Standard Spong acrobot, RK4 integration, `dt = 0.02` with configurable substeps  
- **Network:** MLP (64, 64) with tanh activations, output scaled to `[-100, 100]` Nm  
- **Training:** Differentiable simulation, backpropagation through the RK4 rollout, Adam optimizer  

## Code Changes Made

### New files
| File | Description |
|------|-------------|
| `scripts/run_swingup_experiments.py` | Experiment runner with evaluation harness |

### Modified files
| File | Key changes |
|------|-------------|
| `adaptive_clf/acrobot.py` | Added `n_substeps` to `rk4_step`; added `acrobot_energy` function |
| `adaptive_clf/configs.py` | Added `n_substeps`, `bptt_window`, `lqr_blend`, `lqr_V_threshold`, `lqr_temperature`, `R_energy` to `RolloutConfig`; increased velocity weights in `Q_track` from 0.1 to 5.0 |
| `adaptive_clf/rollout.py` | Added `lqr_blend_control`, `sample_mixed_batch`; integrated BPTT truncation (`stop_gradient`); NaN-safe state clipping; energy-shaping stage cost |
| `adaptive_clf/train.py` | Multi-phase curriculum; per-phase horizon/LR/sampling/energy; `_sanitize_grads` for NaN/inf gradients; gradient clipping |
| `adaptive_clf/__init__.py` | Exported new symbols |
| `scripts/eval_swingup.py` | LQR blend support in evaluation |

## Experiments Run (~30 runs total)

### Phase 1: Baseline + LQR + CLF combinations

| Experiment | Approach | Result |
|-----------|----------|--------|
| Baseline (no LQR, no CLF) | NN only, fixed horizon=200 | Loss diverges immediately (NaN gradients) |
| LQR blend | Sigmoid blend with P_lqr-based switching | LQR region of attraction only ~0.01-0.02 rad; NN never learns to reach it |
| CLF enabled | Lyapunov projection on top of NN | No improvement; CLF constraint not helpful when base policy is unstable |
| LQR + CLF + curriculum | All three combined | No convergence |

**Key finding:** The LQR controller, while theoretically stabilizing the linearized system, has an extremely tiny region of attraction (~0.01 rad, <1 degree) on the nonlinear acrobot at the training `dt`. This makes LQR blending ineffective for bridging the gap between swing-up and stabilization.

### Phase 2: Gradient explosion diagnosis

**Root cause identified:** Backpropagating through >50 steps of acrobot dynamics produces gradient norms of 10^4 -- 10^17, making updates either zero (after clipping) or destructive.

**Mitigations implemented:**
- `_sanitize_grads`: replaces NaN/inf with 0  
- `jnp.nan_to_num` on states and losses  
- Gradient clipping via `optax.clip_by_global_norm`  
- RK4 sub-stepping (`n_substeps=4`) for numerical stability  
- BPTT truncation via `jax.lax.stop_gradient` every N steps  

### Phase 3: Short-horizon + progressive curriculum

Best approach found. Key idea: start training with very short rollout horizons (5 steps = 0.1s) where gradients are meaningful, then gradually increase.

| Phase | Region | Horizon | Loss (start → end) | Gradient norms |
|-------|--------|---------|---------------------|----------------|
| 1 (0.003 rad) | 5 steps | 2130 → 312 | Manageable (~10^4) |
| 2 (0.005 rad) | 10 steps | 1805 → 0.03 | Small (~10) |
| 3 (0.01 rad) | 15 steps | 0.03 → 1.4 | Small |
| 4 (0.02 rad) | 25 steps | 1.2 → 0.2 | Small |
| 5 (0.05 rad) | 40 steps | 4.9 → 4.6 | Moderate (~10^2) |
| 6 (0.1 rad) | 60 steps | 117 → 107 | Exploding again (~10^8) |
| 7+ | 100+ steps | ~600-1900 | Unusable |

**The policy learns excellent local stabilization** in phases 2-5 (loss near zero, terminal norm < 0.1). But the transition to region > 0.1 rad and horizon > 60 always causes divergence.

### Phase 4: Catastrophic forgetting

Even with **mixed sampling** (geometric mixture of small-to-large initial conditions), later phases destroy the near-upright stabilization learned in earlier phases. Evaluation of the medium run (1200 steps):

| Initial condition | |x_T| (NN only) | |x_T| (NN + LQR@0.08) |
|-------------------|----------------|------------------------|
| hang-down (-pi) | 3.08 | 3.08 |
| horizontal (-pi/2) | 3.88 | 3.88 |
| near 0.01 rad | 22.88 | 22.86 |
| near 0.3 rad | 5.45 | 5.45 |

The NN actively destabilizes from near-upright after training on larger regions. LQR blending at eval time doesn't help because the system never reaches the LQR threshold.

### Phase 5: Energy shaping

Added `(E(x) - E_upright)^2` as a stage cost to provide gradient signal for swing-up. Phase 8 loss improved (2493 → 760) but evaluation still shows |x_T| > 3 from all ICs.

## Key Technical Findings

1. **Gradient explosion is the fundamental bottleneck.** The acrobot's fast unstable modes (open-loop eigenvalues with large positive real parts) cause gradients to grow exponentially through the simulation. Even 60 steps (1.2s) at dt=0.02 produces unusable gradients.

2. **Short-horizon training works locally.** Within ~0.05 rad of the upright equilibrium and horizons of 10-40 steps, differentiable simulation gives clean gradients and the policy converges to near-zero loss.

3. **Swing-up requires long horizons that break differentiable simulation.** The acrobot needs 3-5 seconds of coordinated swinging to reach the upright from hanging down. Gradients through that many steps of chaotic dynamics carry no useful information.

4. **Catastrophic forgetting resists simple mitigations.** Mixed sampling, curriculum, and replay of small-IC data all fail because gradient magnitudes from diverged trajectories (large ICs) overwhelm the signal from converged trajectories (small ICs).

5. **LQR has an impractically small region of attraction** for the nonlinear acrobot with these parameters (~0.01-0.1 rad depending on dt and Q/R tuning).

## What Would Be Needed to Make Acrobot Work

The problem is well-studied in the control literature. Likely approaches:

- **Sampling-based RL** (PPO, SAC) instead of analytic gradients through simulation -- avoids the gradient explosion entirely
- **Shooting / collocation methods** (iLQR, direct transcription) -- optimizes open-loop trajectories that are then distilled into a policy
- **Value function as terminal cost** -- train a critic V(x) and use it as terminal cost for short-horizon differentiable rollouts (essentially actor-critic)
- **Energy pumping controller** (analytical, not learned) for the swing-up phase, combined with the NN or LQR for stabilization near the top

## Decision

The acrobot's highly nonlinear, chaotic dynamics make it a poor fit for pure differentiable-simulation-based policy learning without more sophisticated machinery. **Switching to the cart-pole**, which has milder dynamics and where differentiable simulation has already shown promising results (existing `train_cartpole.py` with successful training runs).

## Hardware Note

The machine has an **NVIDIA RTX 5080 GPU** (16 GB, CUDA 12.8) available. Previous runs were inadvertently constrained by sandbox restrictions that blocked GPU access. Future experiments should use `required_permissions: ["all"]` or run outside the sandbox to leverage GPU acceleration.
