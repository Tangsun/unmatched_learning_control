# Cart-Pole Swing-Up via Differentiable Simulation -- Experiment Report

**Date:** 2026-03-06
**Status:** Working -- hybrid LQR+NN policy achieves reliable swing-up

## Goal

Train a neural network control policy to swing up and stabilize a cart-pole system from arbitrary initial conditions (including hanging down at theta=pi), using differentiable simulation and backpropagation through RK4 rollouts.

## System Description

- **State:** `x = [x_cart, theta, x_cart_dot, theta_dot]` where theta=0 is upright, theta=pi is hanging
- **Control:** Horizontal force on the cart, clipped to [-10, 10] N
- **Dynamics:** Standard cart-pole (mc=1.0, mp=0.1, l=0.5), RK4 integration at dt=0.02
- **Uncertainty model:** Cart friction `tau_f = a_true * x_cart_dot` (not used in current experiments, a_true=0)
- **Network:** MLP (64, 64) with tanh activations, output scaled via `u_max * tanh(raw)` then clipped
- **Observation:** 5D: `[x_cart, sin(theta), cos(theta)-1, x_cart_dot, theta_dot]` (zero-centered at upright)
- **Training:** Differentiable simulation, backprop through 200-step RK4 rollout, Adam optimizer

## Code Structure

| File | Role |
|------|------|
| `adaptive_clf/cartpole.py` | Dynamics in control-affine form, linearization, LQR solver |
| `adaptive_clf/train_cartpole.py` | Training loop, cost functions, curriculum sampling, hybrid policy |
| `adaptive_clf/nn.py` | MLP init/apply, policy wrapper with tanh output scaling |
| `scripts/visualize_cartpole.py` | Comprehensive post-training visualization and analysis |

---

## Experiment Timeline

### Phase 0: Early Acrobot Attempts (Pre-Cartpole)

Before switching to cart-pole, ~30 experiments were run on the acrobot swing-up problem. See `docs/acrobot_progress_2026-03-05.md` for full details. Key takeaway: the acrobot's chaotic dynamics cause gradient explosion after ~60 RK4 steps, making differentiable simulation impractical without additional machinery (RL, shooting methods, or value function critics).

**Decision:** Switch to cart-pole, which has milder dynamics and where gradients through 200 steps remain usable.

---

### Phase 1: Initial Cart-Pole Experiments (Acrobot Codebase)

These early runs used the acrobot training infrastructure adapted for cart-pole. The directory names reflect the acrobot-era naming convention (`s2000` = 2000 steps, `h100` = horizon 100).

#### cost_test / cost_test2
- **Purpose:** Validate cost function computation on cart-pole dynamics
- **Duration:** 30 steps each
- **Result:** cost_test had loss ~10^7-10^8 (broken scaling); cost_test2 fixed to ~4900->3850. Terminal norms ~12-14 in both. Confirmed the pipeline works but training is far too short.

#### swingup_0303_1905 (s2000, h100, bs512, lr=3e-4)
- **Result:** First successful convergence. Loss 300->55, terminal norm 12->5.5 over 2000 steps. Eval shows structured swing-up oscillations. Clean, smooth training curve.
- **Key:** Lower learning rate (3e-4) and moderate batch size (512) gave stable gradients.

#### swingup_0303_1917 (s2000, h100, bs4096, lr=5e-4)
- **Result:** Similar success. Loss 320->48, terminal norm 12->3.5. Larger batch size (4096) improved terminal norm. Some robustness issues visible in eval (divergent trajectories from some ICs).

#### swingup_0303_1923 (s2000, h200, bs500, lr=5e-4)
- **Result:** Noisier training. Loss 465->128, terminal norm 6-14 with spikes. Larger horizon (200 vs 100) made optimization harder.
- **Lesson:** Doubling the horizon increases gradient variance significantly.

#### swingup_0303_1936 (s2000, h100, bs500, lr=5e-4)
- **Result:** Clean convergence, loss 320->51, terminal norm 12->4.5. Comparable to 1905 run. No eval plots saved.

#### swingup_0303_1939 (s2000, h100, bs500, lr=5e-4)
- **Result:** FAILED. Loss diverged 758->935, terminal norm stuck at 16.2-16.6. Feasibility metric 0.11-0.20 (should be 1.0). Gradient norms ~10^10.
- **Diagnosis:** Likely a CLF constraint was enabled that caused infeasibility. Same hyperparameters as 1936 but with different config flags.

#### cartpole_swingup_0303_1956 (ep1000, h200, pool10000, bs500, lr=5e-4)
- **Result:** Best Phase 1 run. 20,000 steps (1000 epochs x 20 steps/epoch). Loss 157->19, terminal cost 122->5, terminal norm 9.9->1.36.
- **Key insight:** Longer training (10x more steps) and larger pool (10,000) dramatically improved final performance.
- **Eval:** Successful swing-up and stabilization. Cart stays within +/-0.6m.

#### swingup_0303_2214 (s2000, h100, bs500, lr=5e-4)
- **Result:** Clean convergence, loss 328->52, terminal norm 12->4. Consistent with other h100 runs.

#### swingup_0303_2217 (s2000, h200, bs1000, lr=5e-4)
- **Result:** Noisy convergence, loss 472->138, terminal norm 3-14 with spikes. Large batch didn't compensate for h200 difficulty.

#### swingup_0303_2310 (s2000, h200, bs500, lr=1e-3)
- **Result:** Very noisy. Loss oscillates 50-600 with spikes. Terminal norm 4-16. Eval shows inability to stabilize from many ICs.
- **Lesson:** lr=1e-3 is too aggressive for this problem.

#### swingup_0303_2340 (s2000, h200, bs500, lr=1e-3, region_scale=0.01)
- **Result:** Still noisy despite tighter region. Loss ~109, terminal norm ~12.
- **Lesson:** High LR is the root cause, not region size.

**Phase 1 Summary:**
| Parameter | Good range | Bad range |
|-----------|-----------|-----------|
| Learning rate | 3e-4 to 5e-4 | 1e-3 (too high) |
| Horizon | 100 steps | 200 steps (noisier gradients) |
| Batch size | 500-4096 | - |
| Training steps | 20,000+ | 2,000 (too short) |
| Pool size | 10,000 | 2,000 (less diverse) |

---

### Phase 2: Dedicated Cart-Pole Training (New `train_cartpole.py`)

A new training script was written specifically for cart-pole, introducing the hybrid LQR+NN architecture, curriculum learning, and energy-shaping costs.

#### Smoke Tests (_smoke_test, _smoke_clf, _smoke_phased)
- **_smoke_test:** 5 steps, loss 30->1, terminal norm 19->9. Pipeline works.
- **_smoke_clf:** 3 steps, loss diverges 7->15. CLF variant immediately unstable.
- **_smoke_phased:** 6 steps with phase transitions. Loss spikes to 5000 at phase boundaries. Terminal norm 24->15.
- **Takeaway:** CLF and phase transitions both introduce instability. Pure NN training is most stable.

#### exp_A_baseline (NN-only curriculum, ~160 steps)
- **Result:** Loss fluctuates 100-5000, terminal norm 10-25. No convergence. Baseline curriculum without LQR blending doesn't work for swing-up.

#### test_balance (balance task, ~20 steps)
- **Result:** Loss 3.5->0.5, clean convergence. Balance-only task is easy and converges quickly.
- **Confirms:** The training pipeline works well for the simpler task.

#### test_swingup (swing-up, ~20 steps)
- **Result:** Loss drops initially then spikes at step 15-18. Terminal norm increases 5->12. Too short and no curriculum for swing-up.

---

### Phase 3: LQR Blending and Curriculum Experiments

#### exp_B_lqr_blend (~200 steps)
- **Approach:** LQR blending without curriculum
- **Result:** Loss spikes to 5000+, terminal norm 5-20. LQR alone doesn't help without progressive training.

#### exp_C_lqr_curriculum (~200 steps)
- **Approach:** LQR + curriculum scheduling
- **Result:** Loss elevated at 1000+, terminal norm 8-22. Marginal improvement over baseline.

#### exp_D_lqr_clf_curriculum (~40 steps)
- **Approach:** LQR + CLF + curriculum
- **Result:** Highly volatile loss 1000-5000, terminal norm 15-19. CLF makes things worse.

#### exp_E_lqr_noclip_curriculum (~200 steps)
- **Approach:** LQR + curriculum, no gradient clipping
- **Result:** Loss 1000+, terminal norm 10-20. Removing clipping doesn't help.

#### exp_F_lqr_clf_noclip_curriculum (~200 steps)
- **Approach:** LQR + CLF + no clipping
- **Result:** WORST RUN. Loss spikes to 10^7, terminal norm >200. Catastrophic failure -- CLF without gradient clipping is completely unstable.

#### exp_G_phased_lqr (~350 steps)
- **Approach:** Phased training with LQR, discrete phase boundaries
- **Result:** Loss ~1000 with spikes at phase boundaries. Terminal norm 7-25. Phase transitions disrupt learning.

#### exp_A_phased_lqr (~300 steps, 5 phases)
- **Approach:** More gradual phased LQR
- **Result:** Better than exp_G. Terminal norm decreases 15->6-7. Phase boundaries still cause disruption but less severe.

#### exp_B_phased_lqr_clf (~500 steps)
- **Approach:** Phased LQR + CLF
- **Result:** Loss >10,000, massive spikes at transitions. Terminal norm >17 at transitions. CLF + phase transitions = worst combination.

#### exp_D_lqr_clf_smallregion (~300 steps)
- **Approach:** CLF with restricted initial condition region
- **Result:** Loss 1000-10,000, terminal norm 11-15. Small region doesn't save CLF.

#### exp_E_lqr_noclf_smallregion (~300 steps)
- **Approach:** No CLF, small region
- **Result:** Terminal norm 15->8, modest improvement. Better than CLF variant.

**Phase 3 Summary:**
- CLF constraints consistently hurt performance on cart-pole (unlike theoretical expectation)
- Phase transitions cause loss spikes; smooth curriculum is better than discrete phases
- LQR blending alone is not enough -- needs to be combined with proper curriculum

---

### Phase 4: Curriculum Refinement

#### swingup_curriculum_v1 (~3000 steps)
- **Approach:** Smooth curriculum expansion over many steps, LQR blending
- **Result:** Loss drops to <5 by step 500, terminal norm 25->4. Best convergence of any curriculum run.
- **Key:** Long, smooth curriculum (not discrete phases) + enough total training steps.
- **Eval:** Successful swing-up from multiple ICs. Pole reaches upright with clean oscillation patterns.

#### test_swingup_v2 (~80 steps)
- **Approach:** Quick test of improved training config
- **Result:** Excellent convergence. Loss 800->5, terminal norm 14->3. Very clean training curve.
- **Key insight:** The improved architecture works even with short training.

---

### Phase 5: NN Architecture Iterations (v2-v5)

Each version iterated on cost weights, observation representation, or network architecture.

#### swingup_v2 (full training, ~6500 steps)
- **Result:** Loss noisy 10-500 throughout, terminal norm 4-6 oscillating. Doesn't converge cleanly despite long training.
- **test_v2_quick:** Loss 300->3, terminal norm 26->3. Quick test converges beautifully but full training destabilizes.
- **Diagnosis:** Training too long without learning rate decay -- the policy oscillates around a good solution.

#### swingup_v3 (full training, ~6500 steps)
- **Result:** Most unstable of all versions. Loss spikes to 500+ frequently, terminal norm 0-14 with huge variance.
- **test_v3_quick:** Loss 200->100, terminal norm 26->4. Decent but plateaus early.
- **Diagnosis:** Architecture change (v3) introduced instability in the optimization landscape.

#### swingup_v4 (full training, ~6500 steps)
- **Result:** Loss ~500->10-20 then oscillates. Terminal norm 2-6, volatile but lower than v3.
- **test_v4_quick:** Loss 200->50, terminal norm 25->5. Smoothest convergence among quick tests.
- **Eval (full):** Pole oscillates forever, never reaches upright. NN-only policy fails at swing-up.
- **Key finding:** Without LQR blending, the NN alone cannot reliably swing up AND stabilize.

#### test_v5_quick (hybrid LQR+NN)
- **Result:** Loss 20->100-150 (slight increase). Terminal norm 6-10, oscillatory.
- **Represents** the first hybrid approach test -- promising but not yet tuned.

---

### Phase 6: Acrobot-Style Curriculum on Cart-Pole

These runs applied lessons learned from the acrobot (short horizon, BPTT, energy shaping) to cart-pole.

#### exp_short_horizon
- **Approach:** Very short rollout horizon, expand gradually
- **Result:** Excellent. Loss drops to ~0 by step 100, terminal norm near 0 throughout. Best loss/norm achieved in any experiment.
- **Limitation:** Only works for small initial conditions. Doesn't generalize to full swing-up.

#### exp_bptt (Backprop Through Time with truncation)
- **Approach:** Stop-gradient every N steps to control gradient explosion
- **Result:** Noisy, loss 100-1000 over 2000+ steps. Terminal norm 6-8 at best, high variance.
- **Lesson:** BPTT truncation helps acrobot but isn't needed for cart-pole (gradients are manageable without it).

#### exp_iterated
- **Approach:** Iterate: train short horizon, extend, repeat
- **Result:** Fastest convergence of any run. Loss->0 by step 50, terminal norm->0. Maintained throughout.
- **Limitation:** Like short_horizon, excels at local stabilization but doesn't address full swing-up from theta=pi.

#### exp_A_mixed_noclf (~300 steps, 6 curriculum phases)
- **Approach:** Mixed sampling across regions, no CLF
- **Result:** Terminal norm 17.5->1-2 by step 50 (great), then degrades to 8-15 in later phases.
- **Key issue:** Catastrophic forgetting -- later phases destroy near-upright stabilization.

#### exp_A_mixed_short_horizon (~1000+ steps, 6 phases)
- **Approach:** Mixed sampling + short initial horizon
- **Result:** Terminal norm 12->1-2 early, then degrades to 2-14 across phases. Same forgetting issue.

#### exp_A_energy_mixed (~400 steps)
- **Approach:** Energy-shaping cost `(E - E_upright)^2` + mixed sampling
- **Result:** Highly volatile. Loss 1000->200 then oscillates 100-1000+. Terminal norm 0.5-14, large variance.
- **Eval (from experiment_summary.pkl):** Mean terminal norm ~29.6 near origin, ~3.4 far. Zero swing-up success rate.

#### exp_C_no_energy_mixed (~400 steps)
- **Approach:** Same as above but without energy cost
- **Result:** Similar volatility. Mean terminal norm ~30.5 near origin, ~3.85 far. Zero swing-up success.
- **Conclusion:** Energy shaping provides marginal improvement in the mixed-sampling regime but doesn't solve the fundamental issue.

---

### Phase 7: Final Hybrid LQR+NN (swingup_hybrid_v1) -- BEST RESULT

#### Architecture
- **Hybrid policy:** `u = alpha * u_LQR + (1-alpha) * u_NN`
- **Blending:** `alpha = sigmoid(8 * (cos(theta) - 0.5))` -- LQR dominates within ~60 degrees of upright, NN dominates elsewhere
- **LQR:** Continuous-time ARE solution linearized at upright, Q=diag(1,10,0.1,0.1), R=0.01
- **NN:** MLP(64,64) with tanh activations, 5D observation input
- **Cost:** Stage cost with angle/position/velocity/control weights + energy shaping (w_energy=0.5) + terminal cost (w_terminal=10)
- **Curriculum:** 50% balance ICs + 50% expanding curriculum ICs over first 50% of epochs

#### Training
- **Config:** 200 epochs, horizon=200 (4s), dt=0.02, pool=2048, batch=64, lr=1e-3, grad_clip=10
- **Steps:** ~6400 total (200 epochs x 32 steps/epoch)
- **Training curve:** Noisy -- loss oscillates 3-300 throughout, never cleanly converges. Moving average settles around ~10-20.
- **Gradient norms:** Generally manageable (1-100), occasional spikes to 1000+.

#### Evaluation Results
- **Settle times** (to within +/-10 degrees of upright):
  - Balance (theta=0.2): 0.06s
  - Moderate (theta=1.5): 1.88s
  - Near-hanging (theta=2.8): 2.04s
  - Swing-up (theta=pi): 1.50s
  - Swing-up (theta=-pi): 1.50s
  - Hard (x=1, theta=pi, v=0.5): 2.04s
- **All test cases succeed.** Pole reaches and stays within +/-10 degrees of upright.
- **Cart position:** Drifts during swing-up (up to +/-1.5m) but returns near zero.
- **Control decomposition:** NN provides the swing-up force (~-7 to -10 N initial push), alpha smoothly transitions to LQR as pole approaches upright, LQR handles final stabilization.

---

## Key Ingredients That Make Cart-Pole Work

### 1. Hybrid LQR+NN Blending (Most Important)
The sigmoid blending `alpha = sigmoid(8*(cos(theta) - 0.5))` solves two problems simultaneously:
- **NN only needs to learn swing-up**, not stabilization -- much easier optimization target
- **No catastrophic forgetting** -- LQR always handles the upright region regardless of what the NN does
- The cart-pole LQR has a large enough region of attraction (~60 degrees) to reliably catch the pole once the NN gets it close

### 2. Curriculum with 50% Balance Mixing
Keeping 50% of each training batch as balance ICs (small angles) ensures the NN doesn't forget that it should aim for the upright. The other 50% gradually expands from near-upright to full range.

### 3. Wrapped Angle Observation
Using `[sin(theta), cos(theta)-1]` instead of raw theta provides:
- No discontinuity at theta=+/-pi
- Zero-centered at the goal (cos(0)-1=0, sin(0)=0)
- Smooth gradient signal everywhere

### 4. Energy-Shaping Cost
The term `w_energy * (E - E_upright)^2` gives gradient signal for swing-up even when the pole is far from upright. Without it, the quadratic angle cost `theta^2` has near-zero gradient at theta=pi (wrapped to pi, squared is just pi^2 -- constant-ish).

### 5. Cart-Pole's Favorable Dynamics
Unlike the acrobot, cart-pole dynamics are mild enough that gradients through 200 RK4 steps remain informative. The system is fully actuated in the horizontal direction and has no chaotic regime.

---

## What Doesn't Work

| Approach | Why it fails |
|----------|-------------|
| **CLF constraints** | Always destabilize training. The Lyapunov decrease condition is too restrictive and creates infeasible optimization problems. |
| **Discrete phase transitions** | Loss spikes at phase boundaries. Smooth curriculum is strictly better. |
| **NN-only (no LQR)** | Can learn swing-up motion but cannot stabilize at upright. The two objectives conflict, causing oscillation in optimization. |
| **High learning rate (1e-3)** | Causes noisy training with large loss spikes. Works only when combined with gradient clipping AND hybrid policy. |
| **Long horizon without curriculum** | Gradient variance too high from the start. Must begin with easy ICs. |
| **BPTT truncation** | Unnecessary for cart-pole (gradients don't explode like acrobot). Adds complexity without benefit. |

---

## Remaining Weaknesses

1. **Noisy training curve:** Loss never cleanly converges even in the best run. The optimization landscape is rough due to the mix of easy (balance) and hard (swing-up) ICs in each batch.
2. **No learning rate schedule:** A cosine or step decay after curriculum ends would likely stabilize late training.
3. **Cart drift:** The cart moves +/-1.5m during swing-up. The cost penalizes this (w_x=0.5) but not enough to keep it centered.
4. **No robustness to uncertainty:** All experiments use a_true=0 (no friction). The control-affine uncertainty model exists in the code but hasn't been tested.
5. **Best-model selection is coarse:** Only tracks loss averaged over batches, which is noisy. A dedicated validation set would give cleaner model selection.

---

## Run Index

| Run | Phase | Approach | Loss (final) | Term Norm | Success? |
|-----|-------|----------|-------------|-----------|----------|
| cost_test | 1 | Cost validation | 83M | 12-13 | N/A |
| cost_test2 | 1 | Cost validation | 3852 | 13-14 | N/A |
| swingup_0303_1905 | 1 | NN, h100, lr=3e-4 | 55 | 5.5 | Partial |
| swingup_0303_1917 | 1 | NN, h100, bs4096 | 48 | 3.5 | Partial |
| swingup_0303_1923 | 1 | NN, h200 | 128 | 6-14 | Partial |
| swingup_0303_1936 | 1 | NN, h100 | 51 | 4.5 | Partial |
| swingup_0303_1939 | 1 | NN+CLF | 935 | 16.2 | No |
| cartpole_0303_1956 | 1 | NN, 20K steps | 19 | 1.36 | Yes |
| swingup_0303_2214 | 1 | NN, h100 | 52 | 4 | Partial |
| swingup_0303_2217 | 1 | NN, h200, bs1000 | 138 | 3-14 | Partial |
| swingup_0303_2310 | 1 | NN, lr=1e-3 | 148 | 4-16 | No |
| swingup_0303_2340 | 1 | NN, lr=1e-3, small region | 109 | 12 | No |
| _smoke_test | 2 | Pipeline test | 1 | 9 | N/A |
| _smoke_clf | 2 | CLF test | 15 (div) | 14 | No |
| _smoke_phased | 2 | Phased test | 5000 | 15 | No |
| exp_A_baseline | 2 | NN curriculum | 100-5000 | 10-25 | No |
| test_balance | 2 | Balance only | 0.5 | low | Yes |
| test_swingup | 2 | Swing-up, no curriculum | spike | 12 | No |
| exp_B_lqr_blend | 3 | LQR blend, no curriculum | 5000+ | 5-20 | No |
| exp_C_lqr_curriculum | 3 | LQR + curriculum | 1000+ | 8-22 | No |
| exp_D_lqr_clf_curriculum | 3 | LQR+CLF+curriculum | 1000-5000 | 15-19 | No |
| exp_E_lqr_noclip_curriculum | 3 | LQR+curriculum, no clip | 1000+ | 10-20 | No |
| exp_F_lqr_clf_noclip | 3 | LQR+CLF, no clip | 10^7 | 200+ | No |
| exp_G_phased_lqr | 3 | Phased LQR | 1000 | 7-25 | No |
| exp_A_phased_lqr | 3 | Phased LQR (gradual) | 1000 | 6-7 | Partial |
| exp_B_phased_lqr_clf | 3 | Phased LQR+CLF | 10000+ | 17+ | No |
| exp_D_lqr_clf_small | 3 | CLF, small region | 1000-10K | 11-15 | No |
| exp_E_lqr_noclf_small | 3 | No CLF, small region | 1000-10K | 8-15 | Partial |
| swingup_curriculum_v1 | 4 | Smooth curriculum | <5 | 4 | Yes |
| test_swingup_v2 | 4 | Quick test | 5 | 3 | Yes |
| swingup_v2 | 5 | NN v2, full | 10-500 | 4-6 | Partial |
| swingup_v3 | 5 | NN v3, full | 500+ | 0-14 | No |
| swingup_v4 | 5 | NN v4, full | 10-20 | 2-6 | No |
| test_v4_quick | 5 | NN v4, quick | 50 | 5 | Yes |
| exp_short_horizon | 6 | Short horizon | ~0 | ~0 | Yes (local) |
| exp_iterated | 6 | Iterated extension | ~0 | ~0 | Yes (local) |
| exp_bptt | 6 | BPTT truncation | 100-1000 | 6-8 | No |
| exp_A_mixed_noclf | 6 | Mixed, no CLF | 100-500 | 1-15 | No |
| exp_A_mixed_short | 6 | Mixed+short horizon | 100-1000 | 2-14 | No |
| exp_A_energy_mixed | 6 | Energy+mixed | 100-1000+ | 0.5-14 | No |
| exp_C_no_energy_mixed | 6 | No energy+mixed | 100-1000 | 1-14 | No |
| exp_A_curriculum_noclf | 6 | Curriculum, no CLF | 100-1200 | 1-13 | Partial |
| exp_B_curriculum_clf | 6 | Curriculum+CLF | 100-300 | 8-16 | No |
| **swingup_hybrid_v1** | **7** | **Hybrid LQR+NN** | **~10-20** | **0.5-2** | **Yes** |

---

## Conclusions

1. **Hybrid LQR+NN is the key architectural decision.** It decouples the swing-up problem (NN) from stabilization (LQR), making each sub-problem much easier to solve.

2. **Smooth curriculum with balance mixing prevents catastrophic forgetting.** The 50/50 split between balance and curriculum ICs is critical.

3. **Differentiable simulation works for cart-pole** (unlike acrobot) because the dynamics are mild enough for gradients to propagate through 200 RK4 steps.

4. **CLF constraints consistently hurt.** Across every experiment where CLF was enabled, training was less stable or completely divergent. The projected gradient approach may work better with a pre-trained policy rather than training from scratch.

5. **The current solution is functional but not polished.** Training is noisy, cart drifts, and robustness to model uncertainty hasn't been tested. These are natural next steps.
