# Parameter Estimation Pipeline

This document describes the observer-based adaptive parameter estimation used
in our CLF-shield framework. The estimator runs online during both training
and evaluation, providing the shield with an uncertainty set `B(a_hat, radius)`
that (after the fix below) is guaranteed to contain the true parameter `a_true`.

Reference: Adetola, DeHaan, Guay, "Adaptive model predictive control for
constrained nonlinear systems", Systems & Control Letters 58 (2009) 320–326.

---

## 1. System Assumption

The plant is control-affine with scalar unknown parameter `a`:

```
x_dot = f(x) + G(x) @ u + Y(x) * a
```

where `f(x)`, `G(x)`, `Y(x)` are known (from `affine_terms_fn`), and `a` is
an unknown constant. The parameter lies in a known bounded set
`a ∈ [a_min, a_max]`.

## 2. State Variables

Defined in `AdaptiveState` (NamedTuple, valid JAX pytree):

| Field    | Shape         | Description                                     |
|----------|---------------|-------------------------------------------------|
| `a_hat`  | scalar        | Current parameter estimate                      |
| `radius` | scalar        | Half-width of uncertainty set `B(a_hat, radius)` |
| `x_hat`  | `(state_dim,)` | State predictor estimate                       |
| `w`      | `(state_dim,)` | Filter state                                   |
| `eta`    | `(state_dim,)` | Auxiliary signal (transient from initial error) |
| `info`   | scalar        | Accumulated `||Y||^2` (diagnostic only)         |

## 3. Initialization

`init_adaptive_state(p, adapt_cfg, state_dim, x0, a_range)`:

- `a_hat(0) = 0` (or midpoint of `[a_min, a_max]` if `a_range` not given)
- `radius(0) = a_range` (or `(a_max - a_min) / 2`)
- `x_hat(0) = x0` so that prediction error `e(0) = x - x_hat = 0`
- `w(0) = 0`, `eta(0) = 0`

The initial ball `B(a_hat(0), radius(0))` must contain `a_true`.

## 4. Observer Dynamics

Each timestep, `adaptive_update_observer` computes:

### 4.1 Prediction Error

```
e = x - x_hat
```

### 4.2 Auxiliary Signal (exponential integrator, exact)

```
eta(t+dt) = eta(t) * exp(-k * dt)
```

Decays the initial transient `eta(0) = e(0)` to zero. After decay,
`e - eta` isolates the component of prediction error due to parameter
mismatch (not initial conditions).

### 4.3 Filter (exponential integrator, exact for constant Y)

```
w(t+dt) = w(t) * exp(-k*dt) + Y(x)/k * (1 - exp(-k*dt))
```

The filter `w` tracks the regressor `Y(x)` with time constant `1/k`.
In steady state, `w ≈ Y(x)/k`. The filtered innovation `w^T(e - eta)`
approximates `||w||^2 * a_tilde` where `a_tilde = a_true - a_hat`.

### 4.4 State Predictor (Euler)

```
x_hat_dot = f(x) + G(x) @ u + Y(x) * a_hat + k * e
x_hat(t+dt) = x_hat(t) + x_hat_dot * dt
```

The `k * e` injection drives `x_hat → x`. The predictor uses the current
`a_hat`, so any mismatch `a_tilde` produces a persistent prediction error
`e` that the filter extracts.

### 4.5 Parameter Update

```
innovation = e - eta
a_hat_dot = gamma * w^T * innovation
a_hat_candidate = a_hat + a_hat_dot * dt
a_hat_candidate = clip(a_hat_candidate, a_min, a_max)
```

### 4.6 Lyapunov-Based Radius Bound (eq 15a-b)

The `info` field stores `V_eη`, a Lyapunov energy (eq 15b).

```
V_eη_candidate = V_eη - gamma * ||innovation||² * dt
V_eη_candidate = max(V_eη_candidate, 0)
r_candidate = sqrt(V_eη_candidate)              (eq 15a: z_θ^eη = √V_eη)
```

Note: `z_θ^eη(0) = √(½ z_0²) = z_0/√2 ≈ 0.707·z_0`, intentionally smaller
than the initial radius `z_0`. The gap `z_0 - z_0/√2 ≈ 0.293·z_0` provides
headroom for the joint acceptance condition.

### 4.7 Joint Acceptance (Algorithm 1 from Adetola et al. 2009)

Only publish `(a_hat, radius, V_eη)` when the new ball is contained in the
old ball:

```
delta_a = |a_hat_candidate - a_hat|
accept = (r_candidate <= radius - delta_a)

a_hat_next   = a_hat_candidate  if accept, else a_hat
radius_next  = r_candidate      if accept, else radius
V_eη_next    = V_eη_candidate   if accept, else V_eη
radius_next  = max(radius_next, radius_floor)
```

`V_eη` is frozen on rejection to stay synchronized with `a_hat`.

## 5. Hyperparameters

| Parameter               | Default | Description                                    |
|-------------------------|---------|------------------------------------------------|
| `observer_k`            | 5.0     | Observer gain (prediction error injection rate) |
| `observer_gamma`        | 5.0     | Adaptation gain for `a_hat` update             |
| `observer_eps_w`        | 0.01    | Floor for `||w||^2` in radius computation      |
| `observer_radius_margin`| 0.01    | Safety margin added to radius estimate         |
| `radius_floor`          | 0.001   | Absolute minimum radius                        |

Eval scripts expose `--observer-k` and `--observer-gamma` for tuning.

## 6. How the Shield Uses the Estimate

The CLF shield enforces:

```
LgV^T u <= -lambda * V - LfV - LyV * a_hat - |LyV| * radius
```

This guarantees `V_dot <= -lambda * V` **if and only if**
`a_true ∈ B(a_hat, radius)`. The proof:

```
V_dot = LfV + LgV^T u + LyV * a_true
     <= LfV + (-lambda*V - LfV - LyV*a_hat - |LyV|*radius) + LyV*a_true
      = -lambda*V + LyV*(a_true - a_hat) - |LyV|*radius
```

When `|a_true - a_hat| <= radius`: `LyV*(a_true - a_hat) <= |LyV|*radius`,
so `V_dot <= -lambda * V`.

---

## 7. Bug History

### v1: Independent Update (original)

```python
a_hat_next = a_hat + a_hat_dot * dt                   # always updates
r_candidate = |a_tilde_est| + margin                   # heuristic point estimate
radius_next = min(radius, r_candidate)                 # monotonic shrink
```

**Two independent bugs**:

1. **`a_hat` and `radius` updated independently** — when `a_hat` jumps,
   the new ball `B(a_hat_new, radius_new)` can fail to contain `a_true`
   because the radius didn't account for the center movement.

2. **`r_candidate` is a heuristic point estimate** — computed as
   `|w^T(e-eta) / ||w||^2| + margin`. This estimates `|a_tilde|` from a
   single-step snapshot of the innovation signal. It can dramatically
   **underestimate** the true error (e.g. when `w` happens to be
   near-orthogonal to the actual error direction).

### v2: Joint Acceptance Only (intermediate fix — still broken)

```python
# Same heuristic radius as v1, but with joint acceptance
accept = (r_candidate <= radius - |delta_a|)
a_hat_next   = a_hat_candidate  if accept  else a_hat
radius_next  = r_candidate      if accept  else radius
```

**Why it still failed**: the joint acceptance condition prevents the ball
from "splitting" (bug 1), but it can't fix a bad radius estimate (bug 2).
Example:

```
radius = 0.5, a_hat = 0, a_true = 0.4
Observer produces a_tilde_est ≈ 0 (wrong!)  →  r_candidate = 0.01
a_hat_candidate = 0.001, delta_a = 0.001
accept = (0.01 <= 0.5 - 0.001) = True  ← passes!
New: a_hat = 0.001, radius = 0.01
But: |a_true - a_hat| = 0.399 >> 0.01  ← a_true is outside!
```

The acceptance condition passed because `delta_a` was tiny, but
`r_candidate` was wrong — the point estimate underestimated by 40x.

### v3: Lyapunov-Based Radius + Joint Acceptance (current)

Replaces the heuristic point estimate with a **provably valid** Lyapunov
energy bound from Adetola et al. 2009.

```python
# Lyapunov energy V_eη (eq 15b)
V_eη_candidate = V_eη - gamma * ||innovation||² * dt
r_candidate = sqrt(max(V_eη_candidate, 0))    # eq 15a: z_θ^eη = √V_eη

# Joint acceptance (Algorithm 1)
accept = (r_candidate <= radius - |delta_a|)
a_hat_next   = a_hat_candidate  if accept  else a_hat
radius_next  = r_candidate      if accept  else radius
V_eη_next    = V_eη_candidate   if accept  else V_eη
```

---

## 8. Why the Lyapunov Radius Bound Works

### The key identity

In continuous time, the filter satisfies `e - eta = w * a_tilde` (from the
observer ODEs, eqs 3-6 of the paper). Therefore:

```
||innovation||² = ||e - eta||² = ||w||² * a_tilde²
```

### Two Lyapunov functions decrease at the same rate

Define `V_θ̃ = ½ a_tilde²` (the true estimation energy, not computable) and
`V_eη` (computable, tracked in `info`):

```
V_θ̃_dot  = -a_tilde * a_hat_dot
          = -a_tilde * gamma * w^T(e - eta)
          = -gamma * a_tilde² * ||w||²

V_eη_dot = -gamma * ||innovation||²
         = -gamma * ||w||² * a_tilde²
```

They decrease at **exactly the same rate**.

### The bound

Since `V_θ̃(0) = ½ a_tilde(0)² ≤ ½ radius_0² = V_eη(0)` (by initialization),
and both decrease at the same rate:

```
V_θ̃(t) ≤ V_eη(t)   for all t ≥ 0
```

Therefore:

```
½ (a_true - a_hat(t))² ≤ V_eη(t)
|a_true - a_hat(t)| ≤ sqrt(2 * V_eη(t))
```

The radius `sqrt(2 * V_eη) + margin` is a valid upper bound on `|a_tilde|`.
Unlike the heuristic point estimate, it **cannot underestimate** — it tracks
the cumulative energy extracted from the innovation signal.

### Why V_eη is frozen on rejection

When the joint acceptance rejects an update, `a_hat` stays frozen, so
`a_tilde` stays constant. But if we continued decreasing `V_eη`, it could
drop below `V_θ̃`, breaking the bound. Freezing `V_eη` on rejection keeps
it synchronized with the published `a_hat`.

### Discretization

In discrete time, V_θ̃ has an O(dt²) positive second-order term that V_eη
doesn't, so V_θ̃ decreases slightly LESS than V_eη per step. This means
V_eη decreases slightly faster → the bound `V_θ̃ ≤ V_eη` could eventually
break. The additive `margin` parameter compensates for this discretization
error.

---

## 9. Guarantees

**Invariant**: `a_true ∈ B(a_hat(t), radius(t))` for all `t >= 0`.

Two-layer defense:

1. **Lyapunov bound**: `radius ≥ sqrt(2 * V_eη) ≥ |a_tilde|`, so the
   radius is always a valid bound on the estimation error (modulo small
   discretization error covered by margin).

2. **Joint acceptance**: `B(a_hat_new, r_new) ⊆ B(a_hat_old, r_old)`,
   so the published ball only contracts. Even if the Lyapunov bound has
   small discretization error, the acceptance condition provides an
   additional geometric safety check.

---

## 10. Data Flow Summary

```
x(t) (measured state)
  │
  ├─→ e = x - x_hat                    (prediction error)
  ├─→ eta_next = eta * exp(-k*dt)       (transient decay)
  ├─→ w_next = w*decay + Y(x)/k*(1-decay)  (filter update)
  ├─→ x_hat_next = x_hat + (f + G@u + Y*a_hat + k*e)*dt  (predictor)
  │
  ├─→ innovation = e - eta
  │     ├─→ a_hat_candidate = a_hat + gamma * w^T * innovation * dt
  │     └─→ V_eη_candidate = V_eη - gamma * ||innovation||² * dt
  │         r_candidate = sqrt(2 * V_eη_candidate) + margin
  │
  └─→ Joint acceptance: accept = (r_candidate <= radius - |delta_a|)
        ├─→ YES: publish (a_hat_candidate, r_candidate, V_eη_candidate)
        └─→ NO:  keep    (a_hat,           radius,      V_eη)
                         │
                         ▼
              Shield uses B(a_hat, radius)
              to compute CLF constraint:
              LgV^T u <= -λV - LfV - LyV*a_hat - |LyV|*radius
```
