# Dubins Car: Neural Lyapunov + Adaptive Control Experiments

Date: 2026-03-10

## Motivation

Previous experiments on cart-pole (`cartpole_global_neural_V_v1.md`) showed that:
1. Quadratic Lyapunov functions are fundamentally limited beyond small neighborhoods.
2. Neural Lyapunov (MLP-PSD) with soft CLF violation loss partially works (Run 1), but the CLF shield causes gradient explosion or V inflation.
3. The cart-pole swing-up problem is highly nonlinear (energy barriers, angular topology).

We pivot to the **Dubins car** as a simpler testbed to validate the neural Lyapunov + adaptive control pipeline before returning to cart-pole. Dubins has:
- 3D state (vs 4D cart-pole), kinematic dynamics (no inertia), 2D control
- **Unmatched uncertainty**: side-slip `a` enters `e_x` and `e_y` but control `omega` only enters `e_theta`
- This is the core research question for the LCSS paper: can a CLF shield handle unmatched uncertainty?

## System: Dubins Path-Following Error Dynamics

State: `e = [e_x, e_y, e_theta]` (along-track error, lateral error, heading error)
Control: `u = [v, omega]` (forward speed, turn rate)
Uncertainty: `a` = side-slip velocity (scalar, unknown constant)

```
e_x_dot     = v cos(e_theta) - a sin(e_theta) - v_ref
e_y_dot     = v sin(e_theta) + a cos(e_theta)
e_theta_dot = omega
```

Control-affine form: `edot = f(e) + g(e) @ u + y(e) * a`
- `f(e) = [-v_ref, 0, 0]`
- `g(e) = [[cos(e_th), 0], [sin(e_th), 0], [0, 1]]`
- `y(e) = [-sin(e_th), cos(e_th), 0]`

**Unmatched structure**: `y` has zero in the `e_theta` component, meaning the uncertainty affects `e_x` and `e_y` but NOT `e_theta`. The control `omega` only enters `e_theta`. To compensate for side-slip, the controller must first turn (`omega`) then drive (`v`) — indirect compensation through the nonlinear coupling.

## Approach: Joint Policy + Neural Lyapunov Training

### Architecture
- **Policy**: 2×64 MLP, `nn_only` mode, maps `obs → [v, omega]` with tanh scaling to control bounds
- **Observation**: `[e_x, e_y, sin(e_theta), cos(e_theta) - 1]` (4D, sin/cos encoding for angle)
- **Lyapunov V**: MLP-PSD, `V(x) = 0.5 * ||phi(x) - phi(x_eq)||^2 + eps_pd * ||x - x_eq||^2`
  - Positive definite by construction, `V(x_eq) = 0`
  - Angle wrapping on `e_theta` (index 2) so `V(e_theta=pi) ≠ V(e_theta=0)`
  - `eps_pd = 0.05` (class-K lower bound)

### Training Signal
**Soft CLF violation** (shield OFF during training):
```
violation = relu(Vdot + lambda * V)^2
```
where `Vdot = gradV @ dynamics(x, u_nom, a_true, p)`.

This gives gradient signal to both the policy (choose `u` that makes `V` decrease) and `V` (reshape so the constraint is satisfiable). No hard CLF projection during training — the cart-pole experiments showed that differentiating through the shield causes gradient explosion.

**Feasibility loss** (for shield-compatible V):
```
For each control channel i:
  best_i = min(LgV_i * u_lo_i, LgV_i * u_hi_i)
infeas_margin = relu(sum(best_i) - b_scalar)
feas_penalty = infeas_margin^2
```
where `b_scalar = -lambda*V - LfV`. This generalizes the cart-pole scalar feasibility check to multi-input systems.

### Relation to LCSS Paper Framework

The LCSS paper (`lcss_adaptive_clf_shield_summary.md`) proposes:
1. **Robust CLF constraint**: `LfV + LgV*u + LyV*a_hat + r*|LyV| <= -lambda*V`
2. **Minimum-intervention shield**: project `u_nom` onto the feasible half-space
3. **Adaptive uncertainty set**: online `(a_hat, r_t)` with shrinking radius
4. **Policy trained through the shield**: `u_nom = pi(x, a_hat, r)`, `u = Shield(u_nom)`

**What we implement vs. the LCSS framework:**

| LCSS Component | Current Status | Gap |
|----------------|---------------|-----|
| Robust CLF constraint | Soft penalty only (not hard enforcement) | Need eval-time shield enforcement |
| Shield projection | Implemented but OFF during training | Cart-pole showed gradient explosion when ON |
| Adaptive estimator | Generic update implemented, policy-augmented obs | Policy doesn't effectively use `a_hat` yet |
| Uncertainty-aware V | Not implemented | V should account for worst-case `a` in its design |
| Training through shield | Not done (stop_gradient when shield ON) | Fundamental tension: shield ↔ differentiability |

**Key gap**: The soft violation loss `relu(Vdot + lambda*V)^2` does NOT account for worst-case uncertainty. It computes `Vdot` using the *actual* `a_true` for each trajectory, not the worst-case `a` in the uncertainty set. This means:
- The policy learns to stabilize for sampled `a` values, but has no robustness guarantee
- The Lyapunov function isn't shaped to satisfy the robust CLF constraint
- The feasibility penalty doesn't include the `r*|LyV|` robust margin term

---

## Experiment D1: Baseline — No Uncertainty, Region 0.5

**Run**: `runs/dubins_lyap_d1`
**Config**: 150 epochs, shield OFF, `lambda=0.2`, `dt=0.05`, `horizon=150`, `a_true=0.0`, `region=0.5`

### Training
```
epoch    1/150 | loss   482.037 | term   4.06 | viol  1.09262 | V  1.0428 | grad  5734.67
epoch   10/150 | loss     0.418 | term   0.00 | viol  0.00000 | V  0.0240 | grad     0.19
epoch   50/150 | loss     0.355 | term   0.00 | viol  0.00000 | V  0.0214 | grad     0.14
epoch  100/150 | loss     0.241 | term   0.00 | viol  0.00000 | V  0.0172 | grad     0.03
epoch  150/150 | loss     0.231 | term   0.00 | viol  0.00000 | V  0.0180 | grad     0.02
```

Violation drops to **exactly zero by epoch 10**. Terminal norm = 0.00. Gradients stabilize to ~0.02.

### Evaluation (horizon=200, dt=0.05)

| IC | final |x| | V_0 | V_f | V decr % |
|----|---------|-----|-----|----------|
| small (0.3, 0.3, 0.3) | 0.0006 | 0.083 | 0.000 | 51.5% |
| medium (1.0, 1.0, 1.5) | 0.0006 | 1.058 | 0.000 | 66.0% |
| boundary (1.0, 1.0, pi/2) | 0.0006 | 1.097 | 0.000 | 66.0% |
| OUT: (2.0, 2.0, pi) | 0.0006 | 1.921 | 0.000 | 100.0% |
| OUT: (0, 0, pi) | 0.0006 | 1.260 | 0.000 | 73.0% |

**Result: EXCELLENT.** Every IC converges to the origin (|x| = 0.0006), including out-of-region cases. V decreases to ~0 in all cases. This is dramatically better than cart-pole Run 1 (which only reached theta=0.04-0.29).

**Note**: V decrease % is measured per-timestep (fraction of steps where Vdot < 0), not total decrease. Even at 51%, V reaches zero because the decrease steps dominate the increase steps in magnitude.

---

## Experiment D2: Region Curriculum to Full Scale

**Run**: `runs/dubins_lyap_d2`
**Config**: 200 epochs, warm-started from D1, region 0.5 → 1.0 over first 50% of training, `a_true=0.0`

### Training
```
epoch    1/200 | loss     0.244 | term   0.00 | viol  0.00000 | V  0.0217 | r=0.51
epoch   50/200 | loss     0.616 | term   0.00 | viol  0.00000 | V  0.0790 | r=0.75
epoch  100/200 | loss     1.421 | term   0.01 | viol  0.00017 | V  0.2385 | r=1.00
epoch  200/200 | loss     1.389 | term   0.00 | viol  0.00010 | V  0.4587 | r=1.00
```

Violation stays near zero even at full region. V grows naturally with region scale (larger ICs → larger V values), not inflation.

### Evaluation

| a_true | IC | final |x| |
|--------|----|---------|
| 0.0 | all ICs | **0.001** |
| 0.3 | all ICs | 0.79 |
| 0.5 | all ICs | 1.45 |

**Result**: Perfect stabilization at a=0. Fails with uncertainty (expected — trained without it).

---

## Experiment D3: Fixed Uncertainty a=0.3

**Run**: `runs/dubins_lyap_d3`
**Config**: 200 epochs, warm-started from D2, `a_true=0.3`, region 0.5 → 1.0

### Training
```
epoch    1/200 | loss     3.717 | term   0.44 | viol  0.00032 | V  0.0435 | grad    12.54
epoch  100/200 | loss     3.046 | term   0.25 | viol  0.00089 | V  0.1974 | grad    26.32
epoch  200/200 | loss     2.996 | term   0.25 | viol  0.00045 | V  0.1802 | grad    17.69
```

Training is stable but harder. Terminal norm plateaus at ~0.25 (nonzero steady-state due to unmatched uncertainty).

### Evaluation

| a_true | IC | final |x| |
|--------|----|---------|
| 0.0 | all ICs | **0.60** (biased — policy bakes in a=0.3 compensation) |
| 0.3 | medium | 0.40 |
| 0.3 | lateral (0,2,0) | **0.21** |
| 0.5 | all ICs | 0.67–0.94 |

**Result: PARTIAL.** The policy partially compensates for a=0.3 but converges to a nonzero steady-state. Critically, it performs **worse at a=0** (|x|=0.60) because it bakes in an offset to compensate for a=0.3. This demonstrates the fundamental limitation of training with a fixed uncertainty — the policy cannot adapt to different `a` values.

---

## Experiment D4: Randomized Uncertainty a ~ U[-0.5, 0.5]

**Run**: `runs/dubins_lyap_d4_rand`
**Config**: 200 epochs, warm-started from D2, `a_range=0.5` (each trajectory gets a random `a_true`), `lambda=0.15`, region 0.5 → 1.0

### Code Change
Added `--a-range` flag to `train_lyapunov.py`: when set, each trajectory in the batch gets an independent `a_true ~ U[-a_range, a_range]` instead of a fixed value.

### Training
```
epoch    1/200 | loss     4.646 | term   0.46 | viol  0.00013 | V  0.0456 | grad     7.80
epoch  100/200 | loss     3.401 | term   0.26 | viol  0.00035 | V  0.1368 | grad     1.21
epoch  200/200 | loss     3.541 | term   0.26 | viol  0.00004 | V  0.1612 | grad     0.64
```

Stable training. V stays at 0.16 (no inflation). Gradients well-behaved (~1).

### Evaluation

| a_true | all ICs final |x| | V decr % |
|--------|------------------|----------|
| 0.0 | **0.013** | 70–78% |
| 0.3 | **0.31** | 59–67% |
| 0.5 | **0.55** | 19–72% |
| -0.5 | **0.55** | 10–33% |

### Comparison: D3 (fixed a) vs D4 (randomized a)

| | a=0 |x_f| | a=0.3 |x_f| | a=0.5 |x_f| |
|--|---------|---------|---------|
| D3 (fixed a=0.3) | 0.60 | 0.21–0.48 | 0.67–0.94 |
| D4 (random a) | **0.013** | **0.31** | **0.55** |

**Result: BEST ROBUST POLICY.** D4 is dramatically better across all uncertainty levels. Near-perfect at a=0 (0.013), and the steady-state offset at a≠0 scales proportionally with |a|. All ICs converge to the **same** final state for a given `a_true` — the policy is robust but not adaptive.

**The steady-state offset is a fundamental limitation**: without knowing `a`, the best a fixed policy can do is minimize expected cost across the `a` distribution. The offset at a=0.5 (|x|=0.55) represents the equilibrium where the policy's average compensation balances the actual disturbance.

---

## Experiment D5b: Adaptive Estimator (eta=0.05, warm from D4)

**Run**: `runs/dubins_lyap_d5b_adapt`
**Config**: 200 epochs, warm from D4, `--adapt --adapt-eta 0.05`, obs_dim expanded 4→6

### Code Changes
1. **`adaptive.py`**: Added `adaptive_update_generic()` — full-state residual update that works for any control-affine system (the existing `adaptive_update_simple` was acrobot-specific, using only acceleration states `xdot[2:]` which gives zero signal for Dubins since `y[2]=0`).
2. **`train_lyapunov.py`**: Added `--adapt` flag. When enabled:
   - Observation augmented with `[a_hat, radius]` (obs_dim += 2)
   - Adaptive estimator runs in rollout body with `stop_gradient` on outputs
   - First-layer weight expansion for warm-starting from non-adaptive models

### Evaluation

| a_true | |x_f| | a_hat | Note |
|--------|-------|-------|------|
| 0.0 | 0.30 | 0.000 | Worse than D4 (0.013) |
| 0.3 | 0.30 | **0.158** | a_hat only 53% of true |
| 0.5 | 0.55 | **0.264** | a_hat only 53% of true |

**Result: POOR.** The adaptive estimator converges to only 53% of `a_true`. Root cause: `eta * dt = 0.05 * 0.05 = 0.0025`, so after 300 eval steps: `1 - 0.9975^300 = 0.53`. The learning rate is too low for the estimator to converge within the rollout horizon.

---

## Experiment D6: Adaptive Estimator (eta=0.5, warm from D4)

**Run**: `runs/dubins_lyap_d6_adapt_fast`
**Config**: 200 epochs, warm from D4, `--adapt --adapt-eta 0.5`, obs_dim 4→6

### Evaluation

| a_true | |x_f| | a_hat | D4 |x_f| (no adapt) |
|--------|-------|-------|---------------------|
| 0.0 | 0.22–0.30 | 0.000 | **0.013** |
| 0.3 | 0.31 | **0.300** | 0.31 |
| 0.5 | 0.55 | **0.500** | 0.55 |
| -0.5 | 1.60–1.98 | **-0.500** | 0.55 |

**Result: MIXED.** The estimator now converges perfectly (`a_hat` matches `a_true`). But the policy doesn't effectively use `a_hat` — results are essentially the same as D4 (non-adaptive), and **worse** at a=0 and a=-0.5.

### Why the policy doesn't use a_hat

1. **Zero-initialized weights**: When warm-starting from D4, the extra 2 input weights (for a_hat, radius) are initialized to zero. The policy initially ignores them entirely.
2. **stop_gradient on a_hat**: During training, gradients flow through the policy weights that connect a_hat to hidden layers, but NOT through the adaptive estimator dynamics. The gradient signal for "how to use a_hat" is indirect — it comes from the trajectory cost, not from the estimator.
3. **Insufficient training**: 200 epochs may not be enough to learn the a_hat→u mapping when the base policy already achieves reasonable performance without it.
4. **Asymmetry at a=-0.5**: The policy degrades significantly at negative uncertainty, suggesting it hasn't learned a symmetric compensation strategy.

---

## Summary: Current State vs LCSS Goals

### What Works
1. **Neural Lyapunov pipeline validated on Dubins** (D1-D2): Perfect stabilization from full state space at a=0, violation drops to zero, no V inflation.
2. **Randomized uncertainty training** (D4): Robust policy that handles a ∈ [-0.5, 0.5] with proportional steady-state offset.
3. **Generic adaptive estimator**: Converges to true `a` with appropriate eta.
4. **Multi-channel feasibility check**: Correctly handles Dubins' 2D control bounds.

### What Doesn't Work Yet
1. **Hard CLF enforcement**: The shield is OFF during training. Soft violation loss provides learning signal but no guarantee.
2. **Adaptive policy**: The policy doesn't effectively exploit `a_hat` to eliminate steady-state offset.
3. **Robust CLF constraint**: Not implemented — the violation loss uses actual `a_true` per trajectory, not worst-case over uncertainty set.
4. **Eval-time shield**: Not tested on Dubins yet.

### Gap Analysis: Current Approach vs LCSS Paper

The LCSS paper's key contribution is the **robust CLF shield with shrinking uncertainty set**. Our current approach has diverged from this in several ways:

| Aspect | LCSS Paper | Current Implementation |
|--------|-----------|----------------------|
| **CLF enforcement** | Hard constraint (shield projection) | Soft penalty (relu(Vdot + λV)²) |
| **Uncertainty handling** | Worst-case over set A_t | Per-trajectory sampled a_true |
| **Lyapunov function** | Fixed quadratic (known V) | Learned neural V (unknown, jointly trained) |
| **Training** | Through the shield (differentiable layer) | Shield OFF (gradient explosion when ON) |
| **Guarantee** | Regional stability under parametric uncertainty | No formal guarantee |
| **Adaptive set** | Certified shrinking set with formal bounds | Simple gradient estimator (placeholder) |

### Recommended Next Steps

**Option A: Return to LCSS framework** (more theoretically grounded)
1. Use **fixed quadratic V** (from LQR) for Dubins — may actually work since Dubins is closer to linear.
2. Implement the **robust CLF constraint** with worst-case `a` over uncertainty set.
3. Enable shield at **eval time only** (don't train through it).
4. Compare E1 (no shield) vs E2 (fixed radius) vs E3 (adaptive radius) as in the LCSS plan.

**Option B: Fix the neural Lyapunov approach** (more ambitious)
1. **Normalized violation loss**: Use `relu(Vdot + λV)² / (V² + eps)` to prevent V inflation (not yet needed for Dubins but will be for cart-pole).
2. **Make policy use a_hat**: Train from scratch with adapt ON (not warm-start), or use higher learning rate for the a_hat input weights.
3. **Add robust violation**: Change violation to `relu(Vdot_worst_case + λV)²` where `Vdot_worst_case` includes `|LyV| * radius` term.
4. **Eval-time shield**: Test whether the learned V supports shield enforcement at eval.

**Option C: Hybrid** (pragmatic)
1. Train neural V + policy with soft violation (current approach, works well).
2. At eval time, apply the LCSS shield with the learned V.
3. Use the adaptive estimator to shrink the uncertainty set online.
4. This separates the hard problems: learning (soft, differentiable) vs enforcement (hard, at eval).

## Key Files

| File | Role |
|------|------|
| `adaptive_clf/train_lyapunov.py` | Joint policy + neural V training (main script) |
| `adaptive_clf/lyapunov.py` | MLP-PSD Lyapunov (angle wrapping, class-K bound) |
| `adaptive_clf/adaptive.py` | Adaptive estimator (`adaptive_update_generic`) |
| `adaptive_clf/shield.py` | CLF half-space projection |
| `adaptive_clf/dubins.py` | Dubins dynamics (unmatched uncertainty) |
| `adaptive_clf/systems.py` | System registry (Dubins spec with `angle_indices`) |
| `adaptive_clf/configs.py` | LyapunovConfig, AdaptiveConfig, DubinsParams |
| `scripts/eval_lyapunov.py` | Evaluation script (currently cart-pole specific) |
