# Cart-Pole CLF Experiments (2026-03-07)

## Goal

Learn a neural network control policy for cart-pole stabilization near the upright equilibrium, with Control Lyapunov Function (CLF) constraints providing a stability certificate.

## System

- **State**: `x = [x_cart, theta, x_cart_dot, theta_dot]`, theta=0 is upright
- **Control-affine form**: `xdot = f(x) + g(x)*u + y(x)*a_true` where `y(x)*a_true` models cart friction
- **Training**: Differentiable simulation through 200 RK4 steps, backprop for policy gradients
- **Region scale**: `region_scale=0.3` means ICs sampled with theta in [-0.94, +0.94] rad (54 deg from upright)

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Horizon | 200 steps (4s) |
| dt | 0.02s |
| Batch size | 64 |
| Pool size | 2048 |
| Learning rate | 5e-4 |
| region_scale | 0.3 |
| lambda_clf | 0.5 |
| Network | 2x64 MLP |

## Step 1: NN-only Baseline (no CLF)

**Run**: `runs/step1_nn_only_r03`
**Mode**: `nn_only` -- pure neural network policy, no CLF shield

### Results
- **Best loss**: 17.5
- **Terminal norm**: 1.5-2.5
- **Training**: Converges smoothly, no stability issues

### Evaluation (region_scale=0.3)
- **Small IC (theta=0.47 rad)**: Oscillates but doesn't converge cleanly. Final theta ~-0.27 rad.
- **Boundary IC (theta=-0.94 rad)**: Fails -- pole keeps spinning through full rotations.
- **IC with velocity (theta=0.75, v=0.3)**: Similar failure -- oscillations diverge.
- **Out-of-region (theta=pi)**: Fails completely as expected.

### Assessment
The NN-only policy at region_scale=0.3 learns *something* but is not reliable even within the training region for larger ICs. The small region doesn't provide enough diverse training signal for robust stabilization.

---

## Step 2a: CLF with Fixed P_lqr

**Run**: `runs/step2a_clf_fixed_r03`
**Mode**: `clf` -- NN policy + CLF half-space projection using P from LQR CARE equation
**Lyapunov**: `V(x) = x^T P_lqr x` (fixed, not learned)

### Results
- **Best loss**: 86.8 (much higher than NN-only due to infeasibility penalty)
- **Terminal norm**: 2.4-2.7
- **CLF feasibility**: 0.07 -> 0.54 over training (starts nearly infeasible, improves but never reaches >60%)

### Evaluation
- **Small IC (theta=0.47 rad)**: Stabilizes well -- best of all three methods. Theta converges to ~0.009 rad.
- **Boundary IC (theta=-0.94 rad)**: Fails -- oscillates, final theta ~-2.9 rad.
- **IC with velocity**: Fails similarly.
- **Out-of-region**: Fails.

### Lyapunov Analysis
- V contours (in theta-thetadot plane) are elongated ellipses from the LQR P matrix
- V decreases for the small IC trajectory but has large constraint violations for larger ICs
- Shield is infeasible (violation > 0) for most states outside a tiny neighborhood of the origin
- The LQR-based P is too conservative: it's derived from linearization, so the CLF constraint `dV/dt <= -lambda*V` is only satisfiable very close to the equilibrium

### Assessment
CLF with fixed P_lqr helps for very small perturbations (better than NN-only for theta=0.47) but is severely limited by the infeasibility of the quadratic Lyapunov function in the nonlinear region.

---

## Step 2b: CLF with Learned P (Cholesky)

**Run**: `runs/step2b_clf_learned_r03`
**Mode**: `clf` -- NN policy + CLF projection, with jointly learned Cholesky-parameterized P matrix
**Lyapunov**: `V(x) = x^T P x` where `P = L L^T` is learned alongside the policy

### Results
- **Best loss**: Similar to Step 2a
- **CLF feasibility**: Similar range to Step 2a
- **Learned P eigenvalues**: Very close to P_lqr -- learning did not significantly reshape V

### Evaluation
- Nearly identical behavior to Step 2a across all ICs
- V contours look almost the same as the fixed P_lqr case

### Assessment
The learned P did not diverge from P_lqr. This is likely because:
1. High infeasibility penalties dominate the loss, making gradient signal for P very noisy
2. The quadratic form `x^T P x` is fundamentally limited -- no symmetric positive definite P can make `dV/dt <= -lambda*V` feasible over a large nonlinear region
3. The Cholesky parameterization may need a different learning rate or warmup

---

## Summary and Key Findings

| Metric | Step 1 (NN-only) | Step 2a (CLF fixed) | Step 2b (CLF learned) |
|--------|-------------------|---------------------|----------------------|
| Best loss | 17.5 | 86.8 | ~86 |
| Feasibility | N/A | 0.07->0.54 | ~0.07->0.54 |
| Small IC (0.47 rad) | Oscillates | Stabilizes | Stabilizes |
| Boundary IC (0.94 rad) | Fails | Fails | Fails |
| Out-of-region (pi) | Fails | Fails | Fails |

### Key Takeaways

1. **Quadratic CLF is too restrictive**: Even with learning, `V = x^T P x` cannot provide a feasible CLF constraint beyond a tiny neighborhood of the upright equilibrium. This is a fundamental limitation of quadratic Lyapunov functions for systems with large-angle nonlinearities.

2. **CLF helps locally**: For small perturbations (theta ~0.5 rad), the CLF shield does improve convergence compared to unconstrained NN. The projection steers the policy toward V-decreasing directions.

3. **Infeasibility dominates training**: With ~50-90% of states infeasible, the CLF penalty overwhelms the actual control objective, making training harder and loss higher.

4. **Learned P doesn't help much**: The quadratic structure is the bottleneck, not the specific P matrix.

## Next Steps

### Step 3: Energy-Based Lyapunov Function

The natural next step is to use a physics-informed Lyapunov candidate:

```
V(x) = (E(x) - E_up)^2 + w * x_cart^2
```

where `E(x) = 0.5 * mp * l^2 * thetadot^2 + mp * g * l * (cos(theta) - 1)` is the pole energy and `E_up = 0` is the energy at the upright equilibrium.

**Why this should work better**:
- Energy-based V naturally respects the pendulum's nonlinear dynamics
- The level sets of (E - E_up)^2 follow the separatrix of the pendulum phase portrait
- Feasibility should be much higher because swinging toward the upright *naturally decreases* this V
- The x_cart^2 term prevents the cart from drifting

### Alternative: Neural Lyapunov (MLP-based V)

If the energy-based approach still has issues, a neural network Lyapunov function `V = phi(x)^T phi(x)` (input-convex or PSD by construction) could learn the right level sets from data.

## Generated Artifacts

Each run directory contains:
- `eval_cartpole.png` -- State/control trajectories for 5 ICs (3 in-region, 2 out-of-region)
- `training_curve.png` -- Loss, terminal norm, and feasibility over training
- `lyapunov_analysis.png` (CLF runs) -- Phase portrait with V contours, V along trajectories, shield violation
- `comparison_cartpole.png` (in `runs/`) -- Side-by-side comparison of all methods
- `cartpole_theta*.gif` -- Animations of cart-pole behavior from different ICs
