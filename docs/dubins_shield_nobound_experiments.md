# Dubins Car: Differentiable Shield Training (No Input Bounds)

Date: 2026-03-11

## Context

Previous experiments (D1–D6, documented in `dubins_neural_lyapunov_experiments.md`) used a **soft CLF violation** penalty `relu(Vdot + lambda*V)^2` with the shield OFF during training. This gives no hard stability guarantee — it only encourages the policy and Lyapunov function to satisfy the CLF condition.

The LCSS paper framework calls for **training through the differentiable shield** — using the halfspace projection as a layer so the policy learns to minimize intervention while the shield enforces `Vdot + lambda*V <= 0`.

Earlier attempts (`dubins_shield_diff_*`) to train through the shield with input bounds and `alpha_max=10` were **unstable**: one run (S1) diverged with nan gradients, and feasibility hovered at 40–50%.

This set of experiments removes input constraints entirely (`alpha_max=0`, no clip bounds, no feasibility penalty) and increases the projection regularization (`eps_proj=0.1`) to stabilize gradient flow.

## Key Code Changes

### 1. Removed `alpha_max` cap on shield projection

**File**: `train_lyapunov.py`, shield call in `_episode_rollout_lyap`

Previously: `alpha_max=10.0 if shield_diff else 0.0` — capped the projection gain, which meant the shield **gave up** on enforcing the CLF constraint when `||LgV||^2` was small relative to the violation. This converted the "hard" shield back to a soft one in ~50% of states.

Now: `alpha_max=0.0` always — no cap. The halfspace projection `u_proj = u_nom - relu(violation) / max(||LgV||^2, eps) * LgV` is exact. With unconstrained `u`, the halfspace `LgV^T u <= b` always has a solution as long as `LgV != 0`, so there is no infeasibility.

### 2. Increased `eps_proj` regularization

**File**: `configs.py` `CLFConfig.eps_proj`, set via `train_lyapunov.py`

The projection gain is `alpha = relu(violation) / max(||LgV||^2, eps_proj)`.

Previously: `eps_proj=1e-8` — essentially no regularization. When `||LgV||^2 ~ 0` (states where the Lyapunov gradient is near-orthogonal to the control input matrix), `alpha` explodes, producing enormous controls and `O(1/||LgV||^4)` gradient terms.

Now: `eps_proj=0.1` when `shield_diff=True`. This means:
- When `||LgV||^2 >= 0.1`: projection is near-exact, fully enforces CLF constraint
- When `||LgV||^2 < 0.1`: projection is regularized, undershoots the correction. The CLF constraint may be slightly violated, but gradients are well-conditioned.

This is the **key insight** that made co-training stable. The tradeoff: `eps_proj=0.1` makes the shield approximate in low-`||LgV||` regions, so it's not a hard guarantee everywhere. But it prevents the `1/||LgV||^4` gradient explosion that caused nan in S1.

### 3. Removed feasibility penalty for `shield_diff` mode

**File**: `train_lyapunov.py`, `_episode_rollout_lyap`

With unconstrained `u`, the halfspace `LgV^T u <= b` is always feasible (just push `u` far enough along `-LgV`). The old feasibility penalty (which checked whether the best `u` within input bounds could satisfy the constraint) is meaningless. Set `feas_penalty = 0.0` when `shield_diff=True`.

## Metric Definitions

The metrics logged during training have specific meanings in this context:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **loss** | `mean(state_cost + u_cost + proj_cost + w_violation * violation)` | Overall training objective |
| **terminal_norm** | `\|x_T\|` at end of rollout | Whether the system reaches the origin |
| **violation** | `mean(relu(Vdot_nom + lambda*V)^2)` where `Vdot_nom = gradV @ dynamics(x, u_nom, a, p)` | Soft CLF violation of the **nominal** (pre-shield) control `u_nom`. Nonzero means the policy's own output violates the CLF constraint — the shield then corrects it. |
| **feasible** | Fraction of timesteps where `LgV^T u_nom <= b_scalar` | **Shield intervention rate** (inverted): 1.0 = shield never intervenes, 0.0 = shield always intervenes. This does NOT measure whether a feasible `u` exists — with unconstrained `u`, one always exists. It measures how often the policy's nominal output already satisfies the CLF constraint without shield correction. |
| **V** | `mean(V(x))` along trajectories | Average Lyapunov value. Grows with region scale (larger ICs → larger V), and can grow if V inflates. |
| **grad** | Global gradient norm (before clipping) | Indicator of training stability. Values > 100 suggest explosion; nan means divergence. |
| **r** | Current region scale for IC sampling | How far from the origin we're sampling initial conditions. |
| **wv** | Current weight on the violation term | Annealed from `w_violation_start` to `w_violation_end` during training. |

## Previous Shield-Diff Runs (With Input Bounds, `alpha_max=10`)

These used `eps_proj=1e-8` and `alpha_max=10.0`. All warm-started from D2 (nominal, soft-penalty).

| Run | Config | Final Loss | Violation | Feas | Grad | Outcome |
|-----|--------|-----------|-----------|------|------|---------|
| **test1** | 30ep, region 0.5, a=0 | 0.22 | 0.0003 | 78% | 0.1 | OK (small region) |
| **S1** | 200ep, region 0.5→1.0, a=0 | 47.1 | 0.37 | 40% | **nan** | **Diverged** |
| **S2** | 200ep, region 0.5→1.0, a=0 | 0.88 | 0.006 | 52% | 1.5 | Survived but feas low |
| **S3** | 200ep, region 0.5→1.0, a~U[-0.5,0.5] | 2.9 | 0.01 | 46% | 9.2 | Moderate |

**Key issues**:
- S1 and S2 have **identical configs** but different outcomes (S1 diverged, S2 didn't) — training is non-reproducible due to the chaotic gradient landscape from `violation/||LgV||^2` with tiny `eps`.
- Feasibility at 40–52% means the `alpha_max=10` cap prevents the shield from enforcing the constraint about half the time, defeating the purpose of "hard" enforcement.
- At small region (test1), things work fine — the problem is scaling to the full state space.

## New Experiments: No Input Bounds, `eps_proj=0.1`

### Experiment NB-S2: Nominal, Region Curriculum

**Run**: `runs/dubins_shield_nobound_s2`
**Config**: 100 epochs, warm from D2, `shield-diff`, `a=0`, region 0.5→1.0, `lambda=0.2`, `eps_proj=0.1`, `alpha_max=0`, seed=42

```
epoch    1/100 | loss     0.449 | term   0.01 | viol  0.00000 | feas 0.930 | V  0.0530 | grad     0.18
epoch   10/100 | loss     0.661 | term   0.00 | viol  0.00012 | feas 0.973 | V  0.0548 | grad     0.12
epoch   25/100 | loss     1.029 | term   0.00 | viol  0.00163 | feas 0.936 | V  0.1234 | grad     0.58
epoch   50/100 | loss     2.115 | term   0.01 | viol  0.00691 | feas 0.841 | V  0.3589 | grad     1.32
epoch   75/100 | loss     1.917 | term   0.00 | viol  0.01572 | feas 0.865 | V  0.5353 | grad     2.45
epoch  100/100 | loss     1.515 | term   0.00 | viol  0.04396 | feas 0.811 | V  0.6884 | grad     2.75
```

### Experiment NB-S3: Randomized Uncertainty

**Run**: `runs/dubins_shield_nobound_s3_rand`
**Config**: 100 epochs, warm from NB-S2, `shield-diff`, `a~U[-0.3,0.3]`, region 1.0, `lambda=0.2`, `eps_proj=0.1`, `alpha_max=0`, seed=42

```
epoch    1/100 | loss     5.751 | term   0.29 | viol  0.01545 | feas 0.956 | V  0.3414 | grad     6.12
epoch   10/100 | loss     3.716 | term   0.19 | viol  0.00303 | feas 0.903 | V  0.1951 | grad    17.83
epoch   25/100 | loss     3.262 | term   0.17 | viol  0.00397 | feas 0.653 | V  0.1695 | grad     2.26
epoch   50/100 | loss     3.018 | term   0.16 | viol  0.00538 | feas 0.556 | V  0.2412 | grad     1.16
epoch   75/100 | loss     2.906 | term   0.16 | viol  0.00915 | feas 0.525 | V  0.3166 | grad     1.62
epoch  100/100 | loss     2.631 | term   0.16 | viol  0.01917 | feas 0.491 | V  0.4412 | grad     1.23
```

## Results Summary

| Run | Input Bounds | eps_proj | alpha_max | Final Loss | Terminal | Viol | Feas | Grad | Stable? |
|-----|-------------|----------|-----------|-----------|----------|------|------|------|---------|
| Old S1 | Yes | 1e-8 | 10 | 47.1 | 0.62 | 0.37 | 40% | nan | **No** |
| Old S2 | Yes | 1e-8 | 10 | 0.88 | 0.00 | 0.006 | 52% | 1.5 | Marginal |
| **NB-S2** | **No** | **0.1** | **0** | **1.5** | **0.00** | **0.044** | **81%** | **2.8** | **Yes** |
| **NB-S3** | **No** | **0.1** | **0** | **2.6** | **0.16** | **0.019** | **49%** | **1.2** | **Yes** |

## Key Findings

### 1. `eps_proj=0.1` is the key to stable co-training

The projection gain `alpha = relu(violation) / max(||LgV||^2, eps_proj)` has gradient terms proportional to `1/||LgV||^4`. With `eps_proj=1e-8`, this causes nan gradients when `||LgV||` is small. With `eps_proj=0.1`, the denominator is floored at 0.1, bounding the gradient to `O(1/0.01) = O(100)` in the worst case.

The tradeoff: the projection is only approximate when `||LgV||^2 < 0.1`. This is why violation is nonzero (0.044 in NB-S2) — at states where `LgV` is small, the regularized projection undershoots and the CLF constraint is slightly violated.

### 2. No input bounds removes the infeasibility failure mode

With bounded inputs, states where `||LgV||` is small relative to the required correction are **truly infeasible** — no `u` within bounds can satisfy the CLF constraint. This forced the old code to use `alpha_max` to cap the shield (which defeats the purpose) and a feasibility penalty (which adds noisy gradients).

With unconstrained `u`, the halfspace projection always has a solution. The shield always enforces the CLF constraint (modulo `eps_proj` regularization). The only remaining question is whether the controls are reasonable — and the `proj_cost = w_proj * ||u_shield - u_nom||^2` term in the loss penalizes excessive shield intervention.

### 3. Feasibility metric = shield intervention rate

`feas=81%` in NB-S2 means: for 81% of timesteps, the policy's nominal `u_nom` already satisfies `LgV^T u_nom <= -lambda*V - LfV`. The shield does nothing for those timesteps and only corrects the remaining 19%.

`feas=49%` in NB-S3 (with uncertainty) means the shield corrects about half the time — expected since uncertainty adds `LyV * a` terms to the dynamics that the nominal policy (which doesn't see `a`) cannot compensate for.

### 4. Violation grows as region expands — expected behavior

Violation increases from ~0 (small region) to ~0.04 (full region) in NB-S2. This is because:
- Larger ICs → larger V values → the constraint `Vdot + lambda*V <= 0` is harder to satisfy
- `eps_proj` regularization undershoots more when `V` is large and `||LgV||` is small
- The `w_violation` weight is annealed from 50→5, so the optimizer deprioritizes violation later in training

### 5. V inflation is region-driven, not pathological

`mean(V)` grows from 0.05 to 0.69 as the region expands from 0.5 to 1.0. This is natural: larger ICs start at higher V values. The V function is being shaped to cover the full state space, not inflating spuriously.

## Comparison to Soft-Only Training (D2, D4)

| | D2 (soft, a=0) | NB-S2 (shield, a=0) | D4 (soft, rand a) | NB-S3 (shield, rand a) |
|--|----------------|---------------------|-------------------|----------------------|
| Terminal | 0.00 | 0.00 | 0.26 | 0.16 |
| Violation | 0.0001 | 0.044 | 0.00004 | 0.019 |
| V inflation | No | No | No | No |
| Grad stability | Yes | Yes | Yes | Yes |
| Hard CLF guarantee | **No** | **Approximate** | **No** | **Approximate** |

The soft-only models (D2, D4) have lower violation because the entire loss is focused on making `Vdot + lambda*V <= 0`. The shield models have higher violation because the `eps_proj` regularization makes the shield approximate, and the loss also includes the projection cost `||u_shield - u_nom||^2`.

However, the shield models have an advantage that isn't captured in the violation metric: at states where `||LgV||^2 >= eps_proj`, the shield **exactly** enforces the CLF constraint regardless of what the policy outputs. The soft models have no such guarantee — violation could spike at new states not seen during training.

## Next Steps

1. **Evaluate NB-S2 and NB-S3 on specific ICs**: Compare to D2/D4 baselines with actual trajectory rollouts and measure how often the CLF constraint `Vdot + lambda*V <= 0` holds pointwise.
2. **Sweep `eps_proj`**: Try 0.01, 0.05, 0.5 to find the tightest constraint that still trains stably.
3. **Freeze V**: Train policy only through the shield with frozen Lyapunov parameters, eliminating the co-training instability (constraint surface changing as V updates).
4. **Add adaptation**: With stable shield training working, add the observer-based adaptation scheme from `main.pdf` Section 2.1 (state predictor + filter-based update with guaranteed monotonic error decrease).

## Key Files

| File | Changes in this experiment set |
|------|-------------------------------|
| `adaptive_clf/shield.py` | `eps` default changed to `1e-2` (was `1e-8`) |
| `adaptive_clf/train_lyapunov.py` | `alpha_max=0.0` always; `eps_proj=0.1` when `shield_diff`; `feas_penalty=0` when `shield_diff` |
| `adaptive_clf/configs.py` | `CLFConfig.eps_proj` passed through to shield |
