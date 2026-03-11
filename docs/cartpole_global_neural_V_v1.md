# Cart-Pole Global Stabilization with Neural Lyapunov (v1)

Date: 2026-03-10

## Goal

Learn a neural network Lyapunov function V(x) jointly with a control policy, such that the CLF constraint `dV/dt <= -lambda * V` certifies global stability of the cart-pole at the upright equilibrium. "Global" means stabilization from any initial condition, not just a small neighborhood.

## Background

Previous experiments (see `cartpole_clf_experiments_2026-03-07.md`) showed:

| Step | Method | Small IC (0.47 rad) | Boundary IC (0.94 rad) |
|------|--------|---------------------|------------------------|
| 1 | NN-only (no CLF) | Oscillates | Fails |
| 2a | CLF + fixed P_lqr | Stabilizes | Fails |
| 2b | CLF + learned P (Cholesky) | Stabilizes | Fails |

**Key finding**: Quadratic Lyapunov `V = x^T P x` is fundamentally limited -- no PD matrix P can make the CLF constraint feasible beyond a small neighborhood of the upright equilibrium. The quadratic form doesn't respect the pendulum's nonlinear structure.

## Approach: Joint Policy + Neural Lyapunov Training

### Architecture

- **Policy**: 2x64 MLP, `nn_only` mode (no hybrid LQR blending)
- **Lyapunov V**: MLP-PSD, `V(x) = 0.5 * ||phi(x) - phi(x_eq)||^2 + eps_pd * ||x - x_eq||^2` where phi is a 2x64 MLP. Positive definite by construction, V(x_eq) = 0.

### Training signal

Instead of the binary infeasibility penalty used in Steps 2a/2b, we use a **smooth CLF violation loss**:

```
violation = relu(dV/dt + lambda * V)^2
```

This gives both the policy and V continuous gradient signal:
- **Policy** learns to choose u that makes V decrease
- **V** learns to reshape itself so that the constraint is satisfiable

### Region curriculum strategy

Rather than training on the full state space from the start (which would overwhelm an untrained V), we start with a small region and expand:

1. **Start small** (region_scale=0.3, theta in [-0.94, 0.94] rad) -- get the CLF working locally
2. **Expand** (region_scale -> 1.0) -- once local stabilization is solid, grow the training distribution

---

## Run 1: Baseline — Soft Violation Only, No Shield (Step 3a)

**Run**: `runs/lyap_nn_r03_v1`
**Config**: 100 epochs, w_violation=10, shield OFF, region=0.3, eps_pd=0.001

### Training

```
epoch    1/100 | loss    25.889 | term   3.31 | viol  0.26131 | V  0.0024 | r=0.30 | grad     28.69
epoch   10/100 | loss    24.523 | term   3.20 | viol  0.03780 | V  0.0073 | r=0.30 | grad     15.34
epoch   20/100 | loss    17.476 | term   2.65 | viol  0.00027 | V  0.0106 | r=0.30 | grad      6.81
epoch   50/100 | loss    14.429 | term   1.94 | viol  0.00004 | V  0.0168 | r=0.30 | grad     15.32
epoch  100/100 | loss    12.754 | term   1.96 | viol  0.00002 | V  0.0151 | r=0.30 | grad     12.92
```

Violation drops 4 orders of magnitude (0.26 -> 0.00002).

### Evaluation (no shield)

| IC | Final theta | Final |x| | Final V | V decr % |
|----|-------------|-----------|---------|----------|
| small theta=0.47 | 0.294 | 1.342 | 0.036 | 80.8% |
| boundary theta=0.94 | 0.210 | 1.840 | 0.033 | 84.7% |
| with velocity | 0.039 | 1.545 | 0.005 | 87.7% |
| OUT: theta=pi | -2.588 | 3.615 | 0.410 | 97.2% |
| OUT: theta=2.0 | 1.940 | 7.618 | 0.072 | 79.2% |

### Assessment

**Within region**: Policy reduces theta but doesn't converge cleanly to zero. Final theta 0.04-0.29 rad. V decreases 81-88% of the time -- CLF constraint largely respected but not tight enough for convergence.

**Out of region**: Fails as expected. theta=pi diverges; theta=2.0 barely moves.

**Best result so far** -- the policy at least moves toward the correct equilibrium (upright).

---

## Run 2: Shield ON Without stop_gradient

**Run**: `runs/lyap_nn_shield_warmstart_v1`
**Config**: 100 epochs, warm-started from Run 1, shield ON, w_violation ramp 30->5

### Idea

Since V decreases ~85% of the time in Run 1, enable the CLF shield to hard-enforce the remaining 15%. Shield projects u_nom onto the half-space `{u : LgV*u <= -lambda*V - LfV}`.

### Result: FAILED — Gradient Explosion

Gradients exploded to 10^11 - 10^17 within the first few epochs.

**Root cause**: Differentiating through the halfspace projection (which contains `relu` and division) over 200 RK4 integration steps compounds gradients catastrophically. The projection introduces sharp nonlinearities at every step, and backprop through 200 steps multiplies these.

### Lesson

Cannot naively differentiate through the CLF shield projection in a long rollout. Need to decouple shield enforcement from gradient computation.

---

## Run 3: Shield + stop_gradient

**Run**: `runs/lyap_shield_ws_stopgrad_v1`
**Config**: 100 epochs, warm-started from Run 1, shield ON with `jax.lax.stop_gradient`, w_violation ramp 30->5

### Idea

Use `stop_gradient` on the shield output: the shield enforces safety in the forward pass (actual trajectory uses projected u), but gradients flow only through the soft violation loss computed on `u_nom`. This prevents gradient explosion while still getting trajectory-level safety.

```python
u_shield, shield_aux = clf_shield(...)
u = clip(jax.lax.stop_gradient(u_shield))  # forward: safe u
# but gradients flow through:
Vdot = gradV @ dynamics(x, clip(u_nom), ...)  # backward: smooth
violation = relu(Vdot + lambda*V)^2
```

### Result: Feasibility Stuck at ~45%

Training was stable (no gradient explosion), but shield feasibility plateaued at ~45%. The soft violation loss and shield feasibility measure different things:
- Violation loss: `relu(Vdot + lambda*V)^2` on `u_nom`
- Shield feasibility: whether `LgV*u <= b` has a solution within input bounds

No gradient signal pushed V to make the half-space constraint satisfiable.

### Lesson

`stop_gradient` prevents explosion but creates a disconnect: the policy optimizes soft violation on `u_nom`, while the shield checks a different constraint. Need explicit gradient signal for V to be "shield-compatible."

---

## Run 4: Shield + stop_gradient + Feasibility Penalty

**Run**: `runs/lyap_shield_feas_v1`
**Config**: 100 epochs, from scratch, shield ON, stop_gradient, feasibility penalty added, w_violation ramp 30->5

### Idea

Add an explicit feasibility penalty to give V gradient signal for making the CLF constraint satisfiable:

```python
# CLF is feasible iff b_scalar + |LgV| * u_max >= 0
b_scalar = -lambda*V - LfV
infeas_margin = relu(-(b_scalar + |LgV| * u_max))
feas_penalty = infeas_margin^2
```

This directly pressures V so that the half-space `{u : LgV*u <= b}` intersects the input bounds `[-u_max, u_max]`.

### Result: Wrong Equilibrium (theta=pi)

Feasibility improved to 66.8% (up from 45%), but the policy converged to the **hanging-down equilibrium** (theta=pi) instead of upright (theta=0).

**Eval (with shield)**:

| IC | Final theta | Final |x| | Final V | V decr % | Feasible % |
|----|-------------|-----------|---------|----------|------------|
| small theta=0.47 | -3.141 | 3.141 | 0.000 | 99.5% | 51.8% |
| boundary theta=0.94 | 3.141 | 3.143 | 0.000 | 99.5% | 49.0% |

### Root cause

Two issues conspired:
1. **eps_pd too small**: With `eps_pd=0.001`, `V(theta=pi) >= 0.001 * pi^2 ≈ 0.01` — nearly zero. The class-K lower bound was too weak to prevent V from having a spurious minimum at theta=pi.
2. **No angle wrapping in Lyapunov diff**: The Lyapunov function computed `diff = x - x_eq` without wrapping theta, so the MLP phi could learn `phi(theta=pi) ≈ phi(theta=0)`, making V(theta=pi) ≈ 0.

### Lesson

The Lyapunov function must respect angular topology. A naive Euclidean diff `x - x_eq` makes theta=pi look close to theta=0 (mod 2pi). Must wrap angular components.

---

## Run 5: Shield + Angle Wrapping + Higher eps_pd

**Run**: `runs/lyap_shield_wrapped_v1`
**Config**: 100 epochs, from scratch, shield ON, stop_gradient, feasibility penalty, angle wrapping, eps_pd=0.1, w_violation ramp 30->5

### Code changes

1. **Angle wrapping in Lyapunov** (`lyapunov.py`):
   ```python
   def _wrap_angle_diff(diff, angle_indices):
       for i in angle_indices:
           diff = diff.at[i].set(arctan2(sin(diff[i]), cos(diff[i])))
       return diff
   ```
   Applied in `lyapunov_value()` before computing V. Now theta=pi gives `wrapped_diff[1] = pi`, not 0.

2. **Higher eps_pd = 0.1**: Guarantees `V(theta=pi) >= 0.1 * pi^2 ≈ 0.987`, preventing V from being near-zero at the wrong equilibrium.

3. **angle_indices=(1,)** added to cartpole system spec and `LyapunovConfig`.

4. **Generic gradient path**: `lyapunov_value_and_grad()` uses `jax.grad` (not analytic) when angle wrapping is active, so the chain rule through `arctan2` is handled automatically.

### Training

```
epoch    1/100 | loss   251.684 | term   3.39 | viol  0.16898 | feas 0.497 | V   2.2629 | r=0.30 | wv=30.0 | grad     10.00
epoch   50/100 | loss    57.814 | term   2.13 | viol  0.27809 | feas 0.833 | V  30.2135 | r=0.30 | wv=17.5 | grad     10.00
epoch  100/100 | loss    76.020 | term   2.70 | viol  0.27087 | feas 0.937 | V 178.9929 | r=0.30 | wv= 5.0 | grad     10.00
```

Feasibility improved significantly: 49.7% → 93.7%. But **V inflated** from 2.3 to 179.0, and violation stayed high (0.27).

### Evaluation

**With shield**:

| IC | Final theta | Final |x| | Final V | V decr % | Feasible % |
|----|-------------|-----------|---------|----------|------------|
| small theta=0.47 | -3.060 | 3.112 | 1.024 | 90.5% | 59.7% |
| boundary theta=0.94 | -3.083 | 3.128 | 0.983 | 91.2% | 51.7% |
| with velocity | -3.045 | 3.092 | 1.001 | 91.2% | 55.5% |
| OUT: theta=pi | -3.078 | 3.103 | 0.954 | 98.7% | 44.5% |
| OUT: theta=2.0 | -3.049 | 3.067 | 0.968 | 91.5% | 55.0% |

**Without shield**:

| IC | Final theta | Final |x| | Final V | V decr % |
|----|-------------|-----------|---------|----------|
| small theta=0.47 | -3.092 | 3.291 | 1.201 | 69.5% |
| boundary theta=0.94 | -3.065 | 3.191 | 1.113 | 66.2% |

### Result: FAILED — V Inflation, Wrong Equilibrium Again

All ICs converge to theta ≈ -pi. The angle wrapping fixed the V topology (theta=pi now has V ≈ 1.0, not 0), but the feasibility penalty caused V to inflate to 179, overwhelming the state cost signal.

### Root cause: V inflation

The feasibility penalty `relu(-(b_scalar + |LgV|*u_max))^2` where `b_scalar = -lambda*V - LfV` incentivizes V to grow large. When V is huge, `-lambda*V` makes `b_scalar` very negative, trivially satisfying the constraint regardless of actual dynamics. The feasibility penalty essentially "cheats" by inflating V rather than reshaping it meaningfully.

With V inflated to 179, the w_violation * violation term (0.27 * 5.0 = 1.35) is dwarfed by the feasibility reward from large V, and the state cost (w_theta=10, theta^2 ≈ 10 for theta=pi) can't compete with the V-shaping losses at scale.

---

## Summary of Failure Modes

| Run | Shield | Key Addition | Outcome | Failure Mode |
|-----|--------|-------------|---------|--------------|
| 1 | OFF | Baseline (soft violation only) | Partial stabilization (theta→0.04-0.29) | V not monotonically decreasing |
| 2 | ON | Differentiate through shield | Gradient explosion (10^17) | Backprop through 200 projected RK4 steps |
| 3 | ON | stop_gradient on shield | Feasibility stuck at 45% | No gradient signal for V satisfiability |
| 4 | ON | + feasibility penalty | Wrong equilibrium (theta=pi) | eps_pd too small + no angle wrapping |
| 5 | ON | + angle wrapping + eps_pd=0.1 | Wrong equilibrium (theta=-pi) | V inflation (V=179) from feasibility penalty |

## Key Lessons

1. **Cannot differentiate through shield projection** over long rollouts — gradient explosion is inevitable. `stop_gradient` is necessary.

2. **Feasibility penalty causes V inflation** — the penalty incentivizes V to grow large, making the CLF constraint trivially satisfiable but meaningless. This is a fundamental issue: any loss that rewards `b_scalar = -lambda*V - LfV` being large will inflate V.

3. **Angular topology matters** — Lyapunov functions for systems with angular states must wrap angle differences to `[-pi, pi]`, and the class-K lower bound (`eps_pd * ||diff||^2`) must be large enough to prevent spurious minima.

4. **Run 1 (soft violation only, no shield) was the most successful** — it at least moved toward the correct equilibrium. The shield and feasibility penalty introduced more problems than they solved.

## Proposed Next Steps

1. **Return to shield-OFF training** with soft violation only (like Run 1), but train longer (200-300 epochs) with the angle wrapping fix and eps_pd=0.1.

2. **Increase state cost weight on theta** to more strongly drive upright stabilization.

3. **Consider normalizing the violation loss**: Use `relu(Vdot/V + lambda)^2` to make the constraint scale-invariant and prevent V inflation.

4. **Add a V magnitude penalty** (`mean(V)^2`) if V still inflates.

5. **Enable shield at eval time only** — once V decreases reliably, the shield can enforce the remaining violations without needing to train through it.

## Key Files

| File | Role |
|------|------|
| `adaptive_clf/train_lyapunov.py` | Joint policy + neural V training |
| `adaptive_clf/lyapunov.py` | MLP-PSD Lyapunov function (angle wrapping, class-K bound) |
| `adaptive_clf/shield.py` | CLF half-space projection |
| `adaptive_clf/configs.py` | LyapunovConfig with angle_indices, eps_pd |
| `adaptive_clf/systems.py` | System specs with angle_indices |
| `scripts/eval_lyapunov.py` | Evaluation + plotting script |
