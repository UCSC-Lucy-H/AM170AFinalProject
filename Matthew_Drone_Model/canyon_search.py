# Authors: Matthew Di Fronzo, Kieran Rege, Lucy Han
# Goal: Have the drone avoid hitting canyon walls while flying between waypoints.

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Variables
T_leg = 1.0            # time per leg
N_eval_per_leg = 800   # time samples per leg

safety = 0.1  # margin from each canyon wall

# Obstacles (Canyon walls)
def f_center(x):
    x = np.asarray(x)
    return 1.25 + 0.85*np.sin(1.8*x) + 0.33*np.sin(5.2*x + 0.3)

def f_width(x):
    x = np.asarray(x)
    base = 0.95 + 0.18*np.sin(1.1*x + 1.2)

    pinches = (
        0.30*np.exp(-((x-0.35)/0.20)**2) +  
        0.55*np.exp(-((x-1.00)/0.18)**2) +   
        0.45*np.exp(-((x-1.70)/0.18)**2) +   
        0.45*np.exp(-((x-2.30)/0.20)**2) +   
        0.30*np.exp(-((x-2.95)/0.24)**2)     
    )

    return base - pinches

def f_top(x):
    return f_center(x) + 0.5*f_width(x)

def f_bot(x):
    return f_center(x) - 0.5*f_width(x)

# Gap tester
def chord_min_gap(P0, P1, f_top, f_bot, safety, n=800):
    x0, y0 = P0
    x1, y1 = P1
    t = np.linspace(0.0, 1.0, n)
    x = x0 + t*(x1 - x0)
    y = y0 + t*(y1 - y0)
    top = f_top(x) - safety
    bot = f_bot(x) + safety
    gap = np.minimum(top - y, y - bot)
    return np.min(gap)  # < 0 means straight line collides

def waypoint_gap(P, f_top, f_bot, safety):
    x, y = P
    top = f_top(x) - safety
    bot = f_bot(x) + safety
    return min(top - y, y - bot)  # > 0 means valid

# Waypoints: A B C D E F 
W = np.array([
    [-0.31, 0.4],  # A
    [0.505, 2.05],  # B
    [1.21, 2.09],  # C
    [2.0, 0.60],  # D
    [2.60, 0.84],  # E
    [3.54, 1.28],  # F
], dtype=float)

labels = ["A","B","C","D","E","F"]

print("Waypoint gaps (must be > 0):")
for i, P in enumerate(W):
    g = waypoint_gap(P, f_top, f_bot, safety)
    print(f"{labels[i]}: {g:.4f}")

print("\nStraight-chord gaps (want < 0 to force curvature):")
for i in range(len(W)-1):
    g = chord_min_gap(W[i], W[i+1], f_top, f_bot, safety)
    print(f"{labels[i]}→{labels[i+1]}: {g:.4f}")

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

        # clearance against canyon walls:
        # require top and bottom to be less than y and x
        top = f_top(x)
        bot = f_bot(x)

        gap_top = (top - safety) - y
        gap_bot = y - (bot + safety)
        min_gap = np.min(np.minimum(gap_top, gap_bot))

        # must clear both walls everywhere on the leg
        required_gap = 0.02
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
        print(f"Leg {i}: No p stayed inside canyon. Increase p range, add waypoints, or reduce safety.")
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
plt.plot(x_all, y_all, linewidth=2)
plt.scatter(W[:, 0], W[:, 1], color="black", zorder=3)

# Label waypoints A, B, C, D, E, F
labels = [chr(ord('A') + k) for k in range(len(W))]
for k in range(len(W)):
    plt.text(W[k, 0], W[k, 1], f"  {labels[k]}")

# P values for each curve
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

# plot canyon
x_plot = np.linspace(W[:, 0].min() - 0.2, W[:, 0].max() + 0.2, 600)
plt.plot(x_plot, f_top(x_plot), linewidth=2)
plt.plot(x_plot, f_bot(x_plot), linewidth=2)
plt.plot(x_plot, f_top(x_plot) - safety, "--", linewidth=1.5)
plt.plot(x_plot, f_bot(x_plot) + safety, "--", linewidth=1.5)

plt.axis("equal")
plt.grid(True)
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.title("Multi-leg trajectory through irregular canyon walls")
plt.show()