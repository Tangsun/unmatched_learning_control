"""Check energy-based V CLF feasibility at various cart-pole states."""
import jax
import jax.numpy as jnp
from adaptive_clf.cartpole import cartpole_affine_terms, CartPoleParams

p = CartPoleParams()
mp, l, g = p.mp, p.l, p.g

def energy_V_and_grad(x):
    def V_fn(z):
        E = 0.5 * mp * l**2 * z[3]**2 + mp * g * l * (jnp.cos(z[1]) - 1.0)
        w_cart, w_vel = 1.0, 0.1
        return E**2 + w_cart * z[0]**2 + w_vel * z[2]**2
    V = V_fn(x)
    gradV = jax.grad(V_fn)(x)
    E = 0.5 * mp * l**2 * x[3]**2 + mp * g * l * (jnp.cos(x[1]) - 1.0)
    return V, gradV, E

states = {
    "upright":      jnp.array([0.0, 0.0, 0.0, 0.0]),
    "small_tilt":   jnp.array([0.0, 0.3, 0.0, 0.0]),
    "45deg":        jnp.array([0.0, 0.785, 0.0, 0.0]),
    "horizontal":   jnp.array([0.0, 1.571, 0.0, 0.0]),
    "135deg":       jnp.array([0.0, 2.356, 0.0, 0.0]),
    "hanging":      jnp.array([0.0, 3.14, 0.0, 0.0]),
    "swing_up_1":   jnp.array([0.0, 2.5, 0.0, 2.0]),
    "swing_up_2":   jnp.array([0.0, 1.0, 0.0, 3.0]),
    "near_top_vel": jnp.array([0.0, 0.5, 0.0, -2.0]),
    "hanging_vel":  jnp.array([0.0, 3.0, 0.0, 1.0]),
}

lam = 0.1
header = f"{'State':15s} | {'V':8s} | {'E':8s} | {'LfV':10s} | {'LgV':10s} | {'b=-lV-LfV':10s} | feas?"
print(header)
print("-" * len(header))
for name, x in states.items():
    V, gradV, E = energy_V_and_grad(x)
    f_x, g_x, y_x = cartpole_affine_terms(x, p)
    LfV = float(gradV @ f_x)
    LgV = float(gradV @ g_x)
    V_f = float(V)
    b = -lam * V_f - LfV
    # Best u to satisfy LgV*u <= b
    if LgV > 0:
        best_lhs = LgV * p.u_min
    elif LgV < 0:
        best_lhs = LgV * p.u_max
    else:
        best_lhs = 0.0
    feasible = best_lhs <= b
    feas_str = "YES" if feasible else f"NO (gap={best_lhs - b:.3f})"
    print(f"{name:15s} | {V_f:8.4f} | {float(E):8.4f} | {LfV:10.4f} | {LgV:10.4f} | {b:10.4f} | {feas_str}")
