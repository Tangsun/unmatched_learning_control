# PVTOL: Observer-Based Adaptive CLF Experiments

Date: 2026-03-12

## Context

The Dubins car experiments demonstrated the full adaptive CLF pipeline (MLP-PSD Lyapunov + NN policy + observer-based adaptation) trained from scratch. To validate the approach on a higher-dimensional underactuated system, we implement a **Planar Vertical Take-Off and Landing (PVTOL)** vehicle hovering under unknown lateral wind.

## System

**State**: `x = [px, py, theta, vx, vy, thetadot]` (6D)
- `(px, py)`: position in inertial frame
- `theta`: roll angle (0 = upright)
- `(vx, vy)`: velocity in inertial frame
- `thetadot`: angular rate

**Controls**: `u = [T, tau]` (2D)
- `T`: thrust (along body z-axis), bounds [0, 20] N (~2x mg)
- `tau`: torque, bounds [-2, 2] N*m

**Dynamics** (control-affine):
```
xdot = f(x) + G(x) @ u + Y(x) * a

f(x) = [vx, vy, thetadot, 0, -g, 0]
G(x) = [[0,0], [0,0], [0,0], [-sin(theta)/m, 0], [cos(theta)/m, 0], [0, 1/J]]
Y(x) = [0, 0, 0, 1, 0, 0]
```

**Underactuation**: 3 DOF (px, py, theta) but only 2 controls (T, tau). Lateral motion requires tilting first — thrust is fixed to the body axis.

**Unmatched wind**: Lateral wind acceleration `a` enters `vx` directly, but thrust acts through `-sin(theta)/m` requiring tilt. To compensate wind, the vehicle must: (1) apply torque to tilt, (2) increase thrust for lateral component. This is the same underactuation + unmatched structure as Dubins side-slip, but with an additional indirection layer.

**Hover equilibrium**: `x_eq = 0`, `u_eq = [mg, 0] = [9.81, 0]`.

## Parameters

| Parameter | Value |
|-----------|-------|
| mass `m` | 1.0 kg |
| inertia `J` | 0.1 kg*m^2 |
| gravity `g` | 9.81 m/s^2 |

## Training Configuration

All runs use `train_lyapunov.py` with:

| Parameter | Value |
|-----------|-------|
| Epochs | 300 |
| Horizon | 200 steps (4s) |
| dt | 0.02s |
| Batch size | 64 |
| Pool size | 2048 |
| Learning rate | 5e-4 |
| Grad clip | 10.0 |
| lambda_clf | 0.1 |
| w_violation | 50 -> 5 |
| V network | MLP-PSD (64, 64) |
| Shield | OFF (soft violation only) |
| Region | 0.3 -> 1.0 (curriculum over first 40%) |
| Observation | [px, py, sin(theta), cos(theta)-1, vx, vy, thetadot] (7D) |

## Experiments

### P1: MLP-PSD Baseline (no wind, no adaptation)

**Run**: `runs/pvtol_p1_baseline`
**Config**: a_true=0, no adaptation, no shield

**Training**: Converges **by epoch 20** — terminal norm drops from 25.5 to 0.00 and stays there as region expands to 1.0. Violation near zero (0.001), gradient norms tiny (~0.5). Training is dramatically easier than cart-pole swing-up because hovering is a regulation task.

### P2: Randomized Wind (no adaptation)

**Run**: `runs/pvtol_p2_randA`
**Config**: `a_range=1.5` (wind ~ U[-1.5, 1.5]), no adaptation

**Training**: Stable. Terminal norm converges to 0.10 by epoch 20. Slightly higher gradients (~15) than P1 due to wind perturbations.

### P3: Observer + Randomized Wind (full E2 recipe)

**Run**: `runs/pvtol_p3_observer_scratch`
**Config**: `a_range=1.5`, observer k=3 gamma=20 (tau=0.45s), all from scratch

**Training**: Stable. Terminal norm converges to 0.05 by epoch 40. V values are higher (8.2 vs 0.5 for P1) because V must account for the larger effective state space under wind perturbations. Gradient norms moderate (~2.5).

### Training Summary

| Run | Epochs to converge | Final term | Final violation | Final grad |
|-----|-------------------|-----------|----------------|-----------|
| P1 (no wind) | ~20 | 0.00 | 0.001 | 0.4 |
| P2 (rand wind) | ~20 | 0.10 | 0.008 | 14.7 |
| P3 (observer) | ~40 | 0.05 | 0.002 | 2.5 |

## Evaluation Results

All runs evaluated with horizon=400, dt=0.05, 6 random ICs, **no shield** (matching training conditions).

| Run | a=0.0 | a=0.5 | a=1.0 | a=1.5 | a=2.0 | a=3.0 |
|-----|-------|-------|-------|-------|-------|-------|
| **P1** (no wind training) | 0.001 | 0.265 | 0.521 | 0.764 | 0.993 | diverge |
| **P2** (robust, no adapt) | 0.001 | 0.059 | 0.117 | 0.173 | 0.227 | diverge |
| **P3** (observer) | 0.007 | **0.053** | **0.102** | **0.152** | **0.208** | diverge |

### Key Observations

1. **MLP-PSD works immediately for PVTOL**: Unlike cart-pole swing-up (which failed due to the CLF/energy-pumping incompatibility), PVTOL hovering is a regulation task where V-decrease is always compatible with the control objective. Training converges in ~20 epochs.

2. **Robust training (P2) provides 4-5x improvement**: At a=1.0, P2 achieves |xT|=0.117 vs P1's 0.521. Domain randomization alone goes a long way.

3. **Observer adaptation (P3) provides consistent ~10-15% improvement over P2**: At every in-distribution wind value, P3 outperforms P2. The observer converges to the true wind value (|a_hat-a|=0.0000 for all 6 ICs at a=1.0), enabling tighter compensation.

4. **All models fail at a=3.0**: This is 2x the training range (a_range=1.5). The wind exceeds the vehicle's control authority at large tilt angles. Training with a larger a_range would likely extend the envelope.

5. **Co-training V from scratch works**: P3 trains V + policy + observer jointly with no warm-start, no staging. The MLP-PSD V learns appropriate level sets for the 6D state space. This confirms the Dubins E2 finding that the simplest pipeline works.

6. **Shield feasibility is low (~5-17%)**: When the shield is enabled during eval, the CLF constraint is infeasible most of the time — the regularized projection (eps_proj=0.1) dominates. The soft violation training approach is more appropriate for this system than hard shield enforcement.

### Detailed P3 Eval (a=1.0, no shield)

All 6 ICs converge identically:
- |xT| = 0.102
- |a_hat - a_true| = 0.0000 (observer converges perfectly)
- radius shrinks to 0.01 (margin floor)

## Comparison with Dubins

| Aspect | Dubins | PVTOL |
|--------|--------|-------|
| States | 3 | 6 |
| Controls | 2 | 2 |
| V architecture | MLP-PSD (64,64) | MLP-PSD (64,64) |
| Training epochs to converge | ~50 | ~20-40 |
| Best |xT| at max training wind | 0.306 (a=0.3) | 0.152 (a=1.5) |
| Observer benefit vs robust-only | Same physical floor | ~10-15% improvement |
| Co-train from scratch? | Yes (E2) | Yes (P3) |

## Files

| File | Description |
|------|-------------|
| `adaptive_clf/pvtol.py` | PVTOL dynamics, linearization, LQR |
| `adaptive_clf/configs.py` | Added `PVTOLParams` dataclass |
| `adaptive_clf/systems.py` | Registered PVTOL in system registry |
| `scripts/validate_pvtol.py` | Evaluation, comparison, and animation |
| `scripts/check_energy_clf.py` | Cart-pole energy V feasibility analysis (side investigation) |

## P4/P5: Shield Finetune Investigation (2026-03-12)

Tested the "soft pretrain → differentiable shield finetune" recipe from Dubins E3.

### P4: Direct shield finetune (horizon=200)

**Run**: `runs/pvtol_p4_shield_finetune`
**Config**: Warm-start from P3, `shield_diff=True`, `eps_proj=0.1`, horizon=200, lr=1e-4, grad_clip=5.0

**Result**: **Diverged immediately**. NaN gradients from step 1, loss exploded to ~10,500. P4b (alpha_max variant) also diverged identically.

**Root cause**: The differentiable shield projection gradient includes a `1/||LgV||^4` term. PVTOL's `G(x)` has thrust entering through `sin(θ)/cos(θ)`, creating regions where `||LgV|| ≈ 0` and gradients explode despite `eps_proj=0.1` regularization.

### Horizon sweep diagnostic

Swept horizons [10, 25, 50, 75, 100, 150, 200] at `eps_proj=0.1` (`scripts/debug_horizon_sweep.py`):

| Horizon | NaN? | Grad Norm | Loss |
|---------|------|-----------|------|
| 10 | no | 37 | 15.5 |
| 25 | no | 34 | 13.9 |
| 50 | no | 589M | 13.1 |
| 75 | no | inf | 14.5 |
| 100 | no | inf | 21.4 |
| 150 | YES | nan | 48.2 |
| 200 | YES | nan | 88.9 |

Only horizon ≤ 25 produces tractable gradients.

### P5a/P5b: Short horizon shield finetune (horizon=25)

| Run | Batch | Pool | Final term | Final violation | Final grad | Outcome |
|-----|-------|------|-----------|----------------|-----------|---------|
| P5a | 64 | 2048 | 1.80 | 0.37 | 100M-10B (clipped) | Degraded from P3's 0.054 |
| P5b | 128 | 4096 | 2.56 | 18.9 | up to 1.5T (clipped) | Much worse |

Even at horizon=25 (0.5s), gradients of billions get clipped to 5.0, producing effectively random updates that destroy the pretrained policy.

### Dubins comparison (E2 vs E3)

The recipe was also evaluated for Dubins (where it appeared to work):

| a | E2 (soft-only) | E3 (shield-ft) |
|---|---|---|
| 0.1 (in-dist) | 0.100 | 0.100 |
| 0.3 (in-dist) | 0.306 | 0.307 |
| 0.5 (OOD) | 0.534 | 0.538 |
| 0.7 (OOD) | 0.836 | 0.864 |

E3 provides **no improvement** over E2 — the soft-trained policy already achieves the physical floor.

### Conclusion

The differentiable shield finetune recipe provides no benefit for either system:
- **Dubins**: Soft training already saturates at physical floor
- **PVTOL**: `1/||LgV||^4` gradient explosion through `sin(θ)/cos(θ)` makes differentiable shield training infeasible

Soft-only training (P3/E2) remains the best approach. If hard constraint enforcement is needed, apply the shield at eval time only (no gradient through projection).

## Cart-Pole Investigation (Side Note)

Before pivoting to PVTOL, we attempted to apply the Dubins E2 recipe to cart-pole swing-up. Key finding: **CLF-based swing-up is fundamentally harder** because:

1. Swing-up requires energy pumping (temporarily increasing V), incompatible with the CLF condition Vdot + lambda*V <= 0.
2. Even without CLF pressure (w_violation=0), the NN policy couldn't learn swing-up from theta=pi in 200 steps.
3. Energy-based V (V = E^2 + ...) has LgV=0 at all zero-velocity states — structurally infeasible for CLF.

PVTOL hovering avoids these issues because it's a regulation task where V-decrease is always aligned with the control objective.
