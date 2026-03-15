# Parameter Estimation Pipeline

This document describes the current observer-based adaptive parameter
estimator used by the CLF shield.

The implementation is intended to match the estimator and set-update logic
from:

Adetola, DeHaan, Guay, "Adaptive model predictive control for constrained
nonlinear systems", Systems & Control Letters 58 (2009) 320-326.

Scope note:
- This document covers the estimator only.
- The controller in this repo is still a learned policy plus CLF shield, not
  the MPC law from the paper.

## 1. System Assumption

The plant is control-affine with scalar unknown parameter `a`:

```text
x_dot = f(x) + G(x) @ u + Y(x) * a
```

where `f(x)`, `G(x)`, and `Y(x)` are known through `affine_terms_fn`, and
`a` is an unknown constant contained in a known initial interval.

## 2. What Was Fixed

The previous observer path had three structural mismatches with the paper:

1. The predictor omitted the `w * a_hat_dot` term, so the paper identity
   `e - eta = w * (a_true - a_hat)` no longer held.
2. The code used only the `z_theta^{e eta}` branch from eq. (15), but not
   the excitation-side branch `z_theta^E` from eq. (16).
3. Rejected set updates froze the whole observer state instead of only
   freezing the published controller-facing set.

The current implementation fixes those issues:

1. The predictor now uses
   `x_hat_dot = f + G u + Y a_hat + w a_hat_dot + k e`.
2. The estimator now tracks both `V_{e eta}` and `Q`, and publishes
   `z_theta = min(z_theta^{e eta}, z_theta^E)` as in eqs. (14)-(16).
3. The internal continuous observer keeps evolving every step. Only the
   published `(a_hat, radius)` pair is gated by the paper's nesting test.

## 3. State Variables

`AdaptiveState` now has two layers of state:

### Published state used by the controller

| Field | Meaning |
|---|---|
| `a_hat` | Published parameter estimate |
| `radius` | Published uncertainty radius |
| `info` | Published `V_{e eta}` value associated with `a_hat` |

The shield only reads these published quantities.

### Internal continuous observer state

| Field | Meaning |
|---|---|
| `x_hat` | Predictor state |
| `w` | Filter state from eq. (3) |
| `eta` | Auxiliary state from eq. (6) |
| `a_hat_internal` | Continuously evolving estimate |
| `info_internal` | Continuously evolving `V_{e eta}` from eq. (15b) |
| `q_internal` | Continuously evolving `Q` from eq. (8) |
| `ve0` | Initial `V_E(t_0)` / `V_{e eta}(t_0)` constant used by eq. (16) |

This split is the key fix. The paper's internal estimator and the controller's
published uncertainty set are not the same object.

## 4. Initialization

`init_adaptive_state(...)` sets:

```text
a_hat(0) = 0                      if a_range is given
a_hat(0) = (a_min + a_max) / 2    otherwise

radius(0) = a_range               if a_range is given
radius(0) = (a_max - a_min) / 2   otherwise
```

Observer state:

```text
x_hat(0) = x(0)
w(0) = 0
eta(0) = 0
```

The Lyapunov/set state is initialized as:

```text
V_{e eta}(0) = 0.5 * radius(0)^2
Q(0) = 0
V_E(0) = V_{e eta}(0)
```

Because `V_{tilde a} = 0.5 * |a_true - a_hat|^2`, the certified radius must be
recovered as `sqrt(2V)`. So the internal paper candidate starts at:

```text
z_theta^{e eta}(0) = sqrt(2 * V_{e eta}(0)) = radius(0)
```

and likewise `z_theta^E(0) = radius(0)`.

Implementation note:
- Some OCR/text extractions of the PDF make eqs. (15a) and (16a) look like
  `z = sqrt(V)`.
- That scaling is inconsistent with Lemma 2's inclusion proof because the paper
  defines `V_{tilde a} = 0.5 * |tilde a|^2`.
- The implementation therefore uses `z = sqrt(2V)`, which preserves
  `|a_true - a_hat| <= radius` when the paper inequalities hold.

The initial published ball must contain `a_true`.

## 5. Continuous-Time Observer

At each step, the estimator uses the measured state `x` and control `u`.

Define:

```text
e = x - x_hat
innovation = e - eta
```

The internal observer ODE matches the paper:

```text
a_hat_dot = gamma * w^T * innovation

x_hat_dot = f(x) + G(x) @ u + Y(x) * a_hat + w * a_hat_dot + k * e
w_dot     = Y(x) - k * w
eta_dot   = -k * eta
V_{e eta}_dot = -gamma * ||innovation||^2
Q_dot     = ||w||^2
```

Implementation note:
- The code integrates this ODE with RK4.
- If `dynamics_fn` and `a_true` are available, the RK4 stages also move the
  measured state `x(t)` along the simulated plant during the sample interval.
  This matches the paper's continuous-time estimator more closely than
  freezing `x` over `dt`.

## 6. Set Update from Eqs. (14)-(16)

After the RK4 step, the code computes the two paper radii:

```text
z_theta^{e eta} = sqrt(2 * V_{e eta})
alpha           = 1 / (1 + gamma * Q)             (scalar case of eq. 9)
V_E             = alpha * V_E(t_0)                (eq. 16b)
z_theta^E       = sqrt(2 * V_E)
z_theta         = min(z_theta^{e eta}, z_theta^E) (eq. 14)
```

The internal candidate set is therefore:

```text
B(a_hat_internal_next, r_candidate)
```

with:

```text
r_candidate = min(z_theta^{e eta}, z_theta^E)
```

## 7. Published Set Update (Algorithm 1)

The controller-facing `(a_hat, radius)` pair is only updated if the new ball
is contained in the old published ball:

```text
delta_a = |a_hat_internal_next - a_hat_published|
accept  = (r_candidate <= radius_published - delta_a)
```

If accepted:

```text
a_hat_next = a_hat_internal_next
radius_next = r_candidate
info_next = V_{e eta, internal next}
```

If rejected:

```text
a_hat_next = a_hat_published
radius_next = radius_published
info_next = info_published
```

Critically:
- rejection does not freeze `x_hat`, `w`, `eta`, `a_hat_internal`,
  `info_internal`, or `q_internal`
- it only freezes the published set seen by the shield

That is the intended hybrid structure from the paper.

## 8. What the Shield Uses

The robust CLF shield uses the published ball only:

```text
LgV^T u <= -lambda * V - LfV - LyV * a_hat - |LyV| * radius
```

So the shield's guarantee is only as good as the published invariant:

```text
a_true in B(a_hat, radius)
```

The estimator fixes above were specifically made to restore that invariant.

## 9. Guarantees and Practical Caveat

### Paper guarantee

Under the paper's continuous-time assumptions, if:

```text
a_true in B(a_hat(t_0), radius(t_0))
```

then the published set remains valid for all later times.

The logic is:

1. `V_{e eta}` and `V_E` each produce valid shrinking parameter bounds.
2. `z_theta = min(z_theta^{e eta}, z_theta^E)` is still valid.
3. Algorithm 1 only publishes nested sets.

### Repo caveat

The repo implementation is still a discrete-time RK4 approximation of the
continuous-time proof. So the paper's theorem is not reproduced in a literal
mathematical sense. What was fixed is the structural mismatch:

- correct predictor dynamics
- correct internal/public split
- correct eq. (14)-(16) radius logic
- correct publication rule

In practice this removes the earlier large containment failures. Any remaining
violations should now be at numerical-discretization scale rather than from
the wrong estimator logic.

## 10. Previous Broken Versions

For reference, the older observer path had the following issues.

### v1: Heuristic point-estimate radius

The first version used a single-step heuristic for the radius and updated
`a_hat` and `radius` independently. This could publish a ball that did not
contain `a_true`.

### v2: Joint acceptance without a valid radius

The second version added a nesting test but still used a bad radius estimate.
This prevented geometric "splitting" of the ball, but could still accept
a grossly underestimated set.

### v3: Current paper-aligned estimator

The current version replaces the heuristic radius with the paper's
`min(z_theta^{e eta}, z_theta^E)` construction and only publishes nested sets,
while letting the internal observer evolve continuously.

## 11. Data Flow Summary

```text
measured x, applied u
  |
  +-> internal observer RK4:
      x_hat, w, eta, a_hat_internal, V_{e eta}, Q
  |
  +-> candidate radius:
      z_theta^{e eta} = sqrt(2 * V_{e eta})
      z_theta^E       = sqrt(2 * alpha * V_E(t_0))
      r_candidate     = min(z_theta^{e eta}, z_theta^E)
  |
  +-> publish only if:
      r_candidate <= radius_published - |a_hat_internal_next - a_hat_published|
  |
  +-> shield uses published:
      (a_hat, radius)
```

That is the current estimator update path implemented in `adaptive.py`.
