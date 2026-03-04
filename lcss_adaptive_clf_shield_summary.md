# Adaptive Learning-Based Control with CLF Shielding (Underactuated, Parametric Uncertainty)
*Working summary of our discussion to date (March 2, 2026).*

This note captures:
1) a realistic **two-week plan** to turn the core idea into an L-CSS submission,
2) a **precise problem formulation** (control-affine, linearly parameterized uncertainty + robust CLF projection),
3) a **first concrete example** strategy using the **Acrobot** (underactuated double pendulum) with a fixed (known) Lyapunov function and a differentiable CLF projection layer used during policy training.

---

## 1) Two-week plan to turn the idea into an L-CSS paper

### Goal
Submit a **6-page** L-CSS letter with one tight story:
> A **minimum-intervention adaptive robust CLF shield** for a learned nominal policy, using an **online uncertainty set** for a **linearly parameterized** unknown parameter, demonstrated on an **underactuated** benchmark.

Avoid over-scoping (e.g., learning the Lyapunov function, claiming global underactuated adaptive control, or claiming safety without CBFs). Keep the paper about a *stability filter/shield* that is:
- closed-form (scalar input case),
- differentiable (usable as a training layer),
- less conservative over time as the uncertainty set shrinks.

### Week 1: lock the technical core and get a working simulation
**Days 1–2: Scope + design choices**
- Pick **one benchmark** (Acrobot upright stabilization), not swing-up.
- Choose the uncertainty to be **linearly parameterized** and practical (e.g., unknown viscous friction).
- Fix a **known Lyapunov function**:
  - Local quadratic Lyapunov from LQR around the equilibrium (default).
- Decide on online adaptive state to feed the policy:
  - $o_t = [x_t, \hat a_t, r_t]$.

**Days 3–4: Theory skeleton (paper-ready)**
- Derive robust CLF inequality with uncertainty set $a\in A_t$.
- Show the shield is a **projection onto a half-space** (closed form).
- State regional conditions / feasibility conditions (important for underactuated systems):
  - handle the case $L_g V(x)\approx 0$,
  - define a region $\Omega_\rho = \{x: V(x)\le \rho\}$ where feasibility holds.
- Write 1–2 clean results:
  - Proposition: robust CLF implies $\dot V\le -\lambda V$ under certified set.
  - Corollary: as $r_t$ shrinks, conservatism and intervention shrink.

**Days 5–7: Implementation**
- Implement:
  - dynamics in control-affine + linear-in-parameter form,
  - fixed $V(x)=x^\top P x$, $\nabla V(x)=2Px$,
  - robust CLF shield as a **differentiable projection layer**,
  - simple adaptive update (first pass) for $(\hat a_t, r_t)$,
  - rollout + loss for training $\pi_\theta$ through the shield.
- Produce first plots:
  - stability vs no shield,
  - intervention magnitude $\|u-u_{\text{nom}}\|$,
  - success rate under param uncertainty.

**Go/No-Go checkpoint by end of Day 7**
Proceed with L-CSS push only if:
- the shield yields a **clear robustness improvement**, and
- adaptive shrinking $r_t$ yields **less intervention** than a fixed conservative $r$.

### Week 2: strengthen evidence + write the 6-page letter
**Days 8–10: Experiments (tight set, not a zoo)**
Run the minimal ablations:
- **E1**: nominal learned policy (no shield).
- **E2**: nominal + robust CLF shield with fixed conservative radius $r$.
- **E3**: nominal + adaptive shrinking-set CLF shield ($\hat a_t, r_t$).
- Optional **E4**: training "with the shield in the loop" reduces intervention.

Metrics (collect all):
- stabilization success rate to a target set around the equilibrium,
- max / integral of $V$ increase (should be controlled),
- average and peak intervention $|u-u_{\text{nom}}|$,
- control effort $\sum u^2$,
- runtime per step (closed-form projection advantage).

**Days 11–12: Paper polish items**
- Write the algorithm (pseudo-code):
  - observe $x_t$, update $(\hat a_t,r_t)$,
  - compute $u_{\text{nom}}=\pi_\theta([x_t,\hat a_t,r_t])$,
  - project $u_{\text{nom}}\to u$ via robust CLF constraint and input bounds.
- Clean figure set (3–4 figures max):
  - trajectories,
  - intervention over time,
  - success rate table,
  - optional: conservatism decreasing with shrinking $r_t$.

**Days 13–14: Write the letter**
Suggested 6-page structure:
1. Introduction + contributions (3 bullets)
2. Problem setup and uncertainty model
3. CLF shield derivation (robustification + projection)
4. Learning objective (train through the shield)
5. Experiments (Acrobot)
6. Conclusion + limitations

**Explicit limitations (good to include)**
- linearly parameterized uncertainty (this paper),
- regional guarantee / feasibility set,
- certified initial uncertainty set required,
- stability, not safety (unless you add CBFs later).

---

## 2) Detailed problem formulation (generic, paper-ready)

### System model
We assume control-affine dynamics with **linearly parameterized uncertainty**:

$$
\dot x = f(x) + g(x)u + y(x)a,
$$

where:
- $x\in\mathbb R^n$ is the state,
- $u\in\mathbb R^m$ is the control input (with bounds $u\in\mathcal U$),
- $a\in\mathbb R^p$ is an unknown constant parameter,
- $y(x)\in\mathbb R^{n\times p}$ is a known regressor mapping.

### Adaptive uncertainty set
An online module maintains a **certified uncertainty set**:

$$
a \in A_t.
$$

For the simplest first version:
- **ball set**: $A_t = \{a: \|a-\hat a_t\|_2 \le r_t\}$,
- or **ellipsoid**: $A_t = \{a: (a-\hat a_t)^\top P_t^{-1}(a-\hat a_t) \le 1\}$.

### Lyapunov function (fixed in the first paper)
Use a known local CLF $V:\mathbb R^n\to\mathbb R_{\ge 0}$,
e.g., quadratic:

$$
V(x)=x^\top P x,\quad P\succ 0,\qquad \nabla V(x)=2Px.
$$

Define a desired decay rate:

$$
\alpha(V) = \lambda_{\text{clf}} V.
$$

### Robust CLF inequality
Define Lie derivatives:

$$
L_fV=\nabla V^\top f,\quad L_gV=\nabla V^\top g,\quad L_yV=\nabla V^\top y.
$$

We enforce:

$$
\max_{a\in A_t}\Big(L_fV(x)+L_gV(x)u + L_yV(x)a\Big)\le -\alpha(V(x)).
$$

For the ball set $A_t=\{a:\|a-\hat a_t\|\le r_t\}$:

$$
\max_{a\in A_t} L_yV(x)a = L_yV(x)\hat a_t + r_t\|L_yV(x)^\top\|_2,
$$

which yields the robust CLF constraint:

$$
L_fV(x)+L_gV(x)u + L_yV(x)\hat a_t + r_t\|L_yV(x)^\top\|_2 \le -\lambda_{\text{clf}}V(x).
$$

### Minimum-intervention CLF projection ("shield")
Given a nominal (learned) action $u_{\text{nom}}$, define:

$$
u^\star(x,t)=\arg\min_{u\in\mathcal U}\|u-u_{\text{nom}}\|^2
\quad \text{s.t. robust CLF inequality.}
$$

- In the **single-input** case ($m=1$), this is **closed-form**:
  it reduces to projecting onto an interval (box ∩ half-space).
- This is differentiable almost everywhere and can be used as a layer inside training.

### Learning problem (train through the shield)
Let the policy condition on adaptive state:

$$
o_t = [x_t,\hat a_t,r_t],\qquad u_{\text{nom}}=\pi_\theta(o_t).
$$

Let $u_t = \text{Shield}(x_t, u_{\text{nom}}, \hat a_t, r_t)$.

We train by minimizing expected rollout cost over initial conditions (and optionally true parameters):

$$
\min_\theta\;
\mathbb E_{x_0\sim\mu_0,\;a^\star\sim\mu_a}
\left[
\sum_{k=0}^{N-1}
\big(
x_k^\top Q x_k
+ \rho_u\|u_k\|^2
+ \rho_{\text{proj}}\|u_k-u_{\text{nom},k}\|^2
+ \rho_{\text{infeas}}\,\mathbf 1[\text{infeasible}]
\big)\Delta t
+ x_N^\top Q_f x_N
\right].
$$

Purpose of terms:
- $x^\top Q x$: regulation to equilibrium,
- $\|u\|^2$: energy/effort,
- $\|u-u_{\text{nom}}\|^2$: encourages the policy to "stay inside" the stable set so the shield intervenes less,
- infeasibility penalty: avoids training on states where control authority cannot satisfy CLF.

---

## 3) Strategy for the first example using Acrobot (underactuated)

### Benchmark task
**Local upright stabilization** of the Acrobot around equilibrium

$$
q_1=\pi,\;q_2=0,\;\dot q_1=\dot q_2=0.
$$

We do **not** do swing-up in the first paper.

State (local coordinates):

$$
x=[q_1-\pi,\; q_2,\; \dot q_1,\; \dot q_2].
$$

### Uncertainty choice (keep it linearly parameterized)
Use unknown shoulder viscous friction $d\ge 0$ (scalar):

$$
\tau_f = [d\,\dot q_1,\;0]^\top,
\qquad d\in[d_{\min},d_{\max}].
$$

This leads to the desired affine form

$$
\dot x = f(x)+g(x)u+y(x)d,
$$

where $u$ is elbow torque (scalar) and $y(x)$ comes from the manipulator inverse mass matrix.

### Lyapunov function
Compute local linearization at the upright equilibrium (nominal friction $d_{\text{nom}}$).
Solve LQR to obtain $P\succ 0$, then:

$$
V(x)=x^\top P x,\qquad \nabla V=2Px.
$$

Use decay rate $\lambda_{\text{clf}}$ (tune conservatively).

### Shield (robust CLF projection)
Maintain $(\hat d_t, r_t)$ and enforce:

$$
L_fV(x)+L_gV(x)u + L_yV(x)\hat d_t + |L_yV(x)|\,r_t
\le -\lambda_{\text{clf}}V(x).
$$

Because input is scalar:
- compute the CLF-feasible interval,
- project $u_{\text{nom}}$ into it,
- also enforce torque bounds $u\in[u_{\min},u_{\max}]$.

### Adaptive update (first implementation)
For a first working experiment (not yet the final certified-set theorem):
- use a simple online scalar estimator update for $\hat d_t$,
- define $r_t$ to shrink with accumulated excitation, e.g. $r_t\propto 1/\sqrt{s_t}$.

Later: replace this with the **certified shrinking uncertainty set** update derived from the observer construction in the draft.

### Training setup
- Policy: MLP $\pi_\theta([x,\hat d,r])$ with bounded output (e.g., `tanh` scaled to torque bounds).
- Rollouts: RK4, time step $h$ and horizon $N$.
- Initial condition distribution $\mu_0$ (local):
  - small perturbations around upright.
- True friction distribution $\mu_d$ (uniform on interval).

Loss:
- state quadratic cost $x^\top Q x$,
- control energy $\rho_u u^2$,
- intervention penalty $\rho_{\text{proj}}(u-u_{\text{nom}})^2$,
- infeasibility penalty if the shield is infeasible.

### Minimal ablation experiments
- E1: policy only (no shield).
- E2: fixed robust radius shield (constant $r$).
- E3: adaptive shrinking radius shield ($r_t$).
Optional:
- E4: training without shield vs training through shield and compare intervention.

Key plots:
- state trajectories and stabilization success,
- Lyapunov $V(t)$ decreasing behavior,
- intervention magnitude over time,
- success rate across random seeds and parameter draws,
- runtime per step (closed form).

---

## Appendix: Implementation artifact (JAX scaffold)
A modular JAX scaffold consistent with this plan was drafted as:
- `acrobot_adaptive_clf_jax.py` (dynamics, Lyapunov modules, policy, shield, rollout loss)

It is organized to allow:
- swapping the policy architecture,
- swapping the Lyapunov function (fixed quadratic → learned PSD MLP),
- swapping the adaptive set update module.
