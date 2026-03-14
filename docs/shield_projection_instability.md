# Shield Projection Instability Analysis

## The Mechanism

The CLF shield enforces `LgV^T u <= b` via halfspace projection:

```
alpha = max(0, LgV^T u_nom - b) / max(||LgV||^2, eps_proj)
u_proj = u_nom - alpha * LgV
```

When `||LgV||^2 < eps_proj` (default 0.1), the denominator is regularized.
The projection only removes a fraction `||LgV||^2 / eps_proj < 1` of the
violation. **The CLF constraint is not actually satisfied**, and the guarantee
`V_dot <= -lambda * V` breaks. V can grow, pushing the state further into
regions where LgV stays small — a positive feedback loop.

The robustness term in the constraint incorporates the adaptive observer:

```
b = -lambda * V - LfV - LyV * a_hat - |LyV| * radius
```

Both `a_hat` and `radius` are updated each step. As the observer converges,
the constraint becomes less conservative. However, if the state has already
diverged into a region where `||LgV|| ≈ 0`, the tighter constraint doesn't
help — the projection still can't enforce it.

## PVTOL

State: `[px, py, theta, vx, vy, thetadot]`, control: `[T (thrust), tau (torque)]`.

```
G(x) = [[0,       0    ],    ← px
         [0,       0    ],    ← py
         [0,       0    ],    ← theta
         [-sinθ/m, 0    ],    ← vx
         [cosθ/m,  0    ],    ← vy
         [0,       1/J  ]]   ← θ̇
```

The two LgV components:

```
LgV[0] = (-sinθ · ∂V/∂vx + cosθ · ∂V/∂vy) / m    (thrust channel)
LgV[1] = (1/J) · ∂V/∂θ̇                             (torque channel)
```

**LgV[0] (thrust) vanishes** when the thrust direction (body z-axis) is
orthogonal to the V-gradient in velocity space:

```
tanθ = (∂V/∂vy) / (∂V/∂vx)
```

At large θ, the vehicle is tilted so far that thrust pushes laterally rather
than correcting the error. The torque channel (LgV[1]) can still rotate the
vehicle back, but this is indirect and slow — θ must change first, then thrust
becomes effective again. During this lag, V grows unchecked.

**Why it cascades**: large wind → vehicle tilts to compensate → if tilt
overshoots or observer is slow → LgV[0] → 0 → shield ineffective → state
diverges → more tilt → stuck.

**LgV[1] (torque)** = `(1/J) · ∂V/∂θ̇` — this only vanishes if V is flat
w.r.t. angular rate, which is unlikely for a well-trained Lyapunov function.

## Dubins Car

State: `[ex, ey, eθ]` (path-following error frame), control: `[v, omega]`.

```
G(x) = [[cosθ, 0],
         [sinθ, 0],
         [0,    1]]
```

The two LgV components:

```
LgV[0] = cosθ · ∂V/∂ex + sinθ · ∂V/∂ey    (speed channel)
LgV[1] = ∂V/∂eθ                             (turn-rate channel)
```

**LgV[0] (speed) vanishes** when the heading direction `(cosθ, sinθ)` is
orthogonal to the V-gradient in position space:

```
tanθ = -(∂V/∂ex) / (∂V/∂ey)
```

**LgV[1] = ∂V/∂eθ** vanishes only if V is flat w.r.t. heading error —
unlikely unless eθ is already near its optimal value.

### Dubins is less vulnerable than PVTOL because:

1. The speed channel LgV[0] depends on `(cosθ, sinθ)` which sweeps a full
   circle — it's only zero at one specific heading angle, and the turn-rate
   channel can quickly rotate away from it.
2. The turn-rate channel LgV[1] is independent of position/heading
   configuration — it depends only on V's sensitivity to heading error.
3. Dubins has 3 states vs PVTOL's 6 — fewer dimensions for the gradient to
   become orthogonal to the control columns.
4. The dubins dynamics are kinematic (no inertia), so corrections are
   immediate — no lag between torque and heading change like PVTOL has between
   torque and thrust direction.

## Root Cause

This is fundamentally a **loss of relative degree / underactuation** issue.
The CLF assumes control authority over V_dot via `LgV^T u`, but at certain
states the control input matrix `G(x)` becomes (near-)orthogonal to `gradV`.
No finite control can enforce V-decrease there. The `eps_proj` regularization
prevents infinite controls but sacrifices the guarantee.

## Potential Mitigations

- **Reduce `eps_proj`**: allows harder projection but risks large control
  spikes.
- **Fallback to policy**: when `||LgV||^2 < eps_proj`, skip the projection
  and let the nominal policy act unshielded.
- **Train V to avoid LgV ≈ 0**: add a penalty on `1/||LgV||^2` during
  training to discourage Lyapunov functions that lose control authority.
- **Input-constrained QP**: replace halfspace projection with a QP that
  respects actuator limits, avoiding the cascading overshoot scenario.
