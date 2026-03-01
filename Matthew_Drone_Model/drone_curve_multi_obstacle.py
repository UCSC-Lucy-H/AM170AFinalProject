# Authors: Matthew Di Fronzo, Kieran Rege, Lucy Han
# Goal: Have the drone avoid multiple obstacles in between several waypoints.

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Variables
T_leg = 1.0            # time per leg
N_eval_per_leg = 800   # time samples per leg

safety = 0.1 # Safety parameter to make sure we visibly clear obstacles

# Obstacles
obstacles = [
    (0.55, 1.05, 0.25),
    (1.60, 1.20, 0.30),
    (2.30, 1.80, 0.20),
]

# Waypoints (Current A to B to C to D)
W = np.array([
    [0.0, 0.0],   # A
    [1.0, 2.0],   # B
    [2.0, 1.0],   # C
    [3.0, 2.5],   # D
], dtype=float)


# Brute force p check
potential_p = []
for val in np.linspace(0.0, 12.0, 241):
    potential_p.append(val)
    potential_p.append(-val)

# time grid per leg
t_eval_leg = np.linspace(0, T_leg, N_eval_per_leg)

def run_one_leg(x0, y0, xT, yT):
    dx = xT - x0
    dy = yT - y0

    L = np.sqrt(dx*dx + dy*dy)
    if L == 0.0:
        raise ValueError("Zero-length leg (two identical waypoints).")

    # Unit normal to direction (dx, dy)
    nx = -dy / L
    ny =  dx / L

    best_p = None
    best_gap = None
    best_sol = None

    for p in potential_p:

        def rhs(t, X):
            # Trajectory equation
            # X(t) = A + h(s)(B-A) + p*b(s)*n,  s = t/T_leg
            # h(s) = 3s^2 - 2s^3  -> h'(s) = 6s - 6s^2
            # b(s) = s^2(1-s)^2  -> b'(s) = 2s - 6s^2 + 4s^3
            s = t / T_leg

            hx = dx * (6*s - 6*(s**2))
            hy = dy * (6*s - 6*(s**2))

            bx = p * nx * (2*s - 6*(s**2) + 4*(s**3))
            by = p * ny * (2*s - 6*(s**2) + 4*(s**3))

            vx = (hx + bx) / T_leg
            vy = (hy + by) / T_leg
            return [vx, vy]

        sol = solve_ivp(rhs, (0, T_leg), [x0, y0],
                        t_eval=t_eval_leg, rtol=1e-9, atol=1e-12)

        x = sol.y[0]
        y = sol.y[1]

        # clearance against ALL obstacles
        min_gap = np.inf
        for (ox, oy, oR) in obstacles:
            d = np.sqrt((x - ox)**2 + (y - oy)**2)
            min_gap = min(min_gap, np.min(d) - (oR + safety))

        # must clear every obstacle
        required_gap = 0.02   # extra beyond (R + safety). Tune: 0.00 to 0.05

        if min_gap >= required_gap:
            if (best_p is None) or (abs(p) < abs(best_p)):
                best_p = p
                best_gap = min_gap
                best_sol = sol

    return best_p, best_gap, best_sol

# Running all time legs
x_all, y_all, t_all = [], [], []
t_offset = 0.0
p_list = []


for i in range(len(W) - 1):
    x0, y0 = W[i]
    xT, yT = W[i+1]

    best_p, best_gap, sol = run_one_leg(x0, y0, xT, yT)

    if sol is None:
        print(f"Leg {i}: No p cleared obstacles. Increase p range or change waypoints.")
        break

    print(f"Leg {i}: {W[i]} -> {W[i+1]} | p = {best_p:.3f} | min gap = {best_gap:.4f}")
    p_list.append(best_p)

    x_leg = sol.y[0]
    y_leg = sol.y[1]
    t_leg = sol.t + t_offset

    # avoid duplicating join point
    if i > 0:
        x_leg = x_leg[1:]
        y_leg = y_leg[1:]
        t_leg = t_leg[1:]

    x_all.append(x_leg)
    y_all.append(y_leg)
    t_all.append(t_leg)

    t_offset += T_leg

# Error if program fails to run
if len(x_all) == 0:
    raise RuntimeError("No legs were successfully simulated.")

x_all = np.concatenate(x_all)
y_all = np.concatenate(y_all)
t_all = np.concatenate(t_all)


# Plot results
plt.figure()
plt.plot(x_all, y_all)
plt.scatter(W[:, 0], W[:, 1], color="black", zorder=3)

# Label waypoints A, B, C, D
labels = [chr(ord('A') + k) for k in range(len(W))]
for k in range(len(W)):
    plt.text(W[k, 0], W[k, 1], f"  {labels[k]}")

# annotate p values on the path
leg_names = [f"{labels[i]}→{labels[i+1]}" for i in range(len(p_list))]
lines = [f"{leg_names[i]}: p = {p_list[i]:.2f}" for i in range(len(p_list))]
txt = "curve p-values\n" + "\n".join(lines)

plt.gca().text(
    0.02, 0.98, txt,
    transform=plt.gca().transAxes,
    va="top", ha="left",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.9)
)

# draw obstacles 
theta = np.linspace(0, 2*np.pi, 300)
for (oxc, oyc, oR) in obstacles:
    ox = oxc + oR*np.cos(theta)
    oy = oyc + oR*np.sin(theta)
    plt.plot(ox, oy, linewidth=2)  

plt.axis("equal")
plt.grid(True)
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.title("Multi-leg trajectory with circular obstacles")
plt.show()