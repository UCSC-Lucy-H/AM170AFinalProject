# Authors: Matthew Di Fronzo, Kieran Rege, Lucy Han
# Goal: Check to see that numerical calculation matches predicted calculations

# Libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Variables
x0, y0 = 0.0, 0.0 # Starting points
xT, yT = 1.0, 2.0 # Ending points
T = 1.0           # Final time
E_H = 1.0         # Energy for hover
xc, yc = 0.55, 1.05   # Obstacle coordinates
R = 0.25              # Obstacle radius


# Bulge/Bump/Curve
# X(t) = A + h(s)(B - A) + pb(s)n, s = T/t
# h(s) = 3s^2 - 2s^3
# b(s) = s^2(1 - s^2)


# Time split for halfway point
T_leg = T / 2.0

dx = xT - x0
dy = yT - y0

# Unit normal to direction (dx, dy)
L = np.sqrt(dx*dx + dy*dy)
nx = -dy / L 
ny = dx / L 

t_eval = np.linspace(0, T, 2000)

# Brute forcing best p
potential_p = []
for val in np.linspace(0.0, 6.0, 121): 
    potential_p.append(val)
    potential_p.append(-val)

best_p = None
best_min_dist = None


for p in potential_p:

    def rhs(t, X):
        if t <= T_leg:
            tl = t
            s = tl / T_leg

            hx = dx * (6*s - 6*(s**2))
            hy = dy * (6*s - 6*(s**2))

            bx = p * nx * (2*s - 6*(s**2) + 4*(s**3))
            by = p * ny * (2*s - 6*(s**2) + 4*(s**3))

            vx = (hx + bx) / T_leg
            vy = (hy + by) / T_leg
        else:
            tl = t - T_leg
            s = tl / T_leg

            hx = (-dx) * (6*s - 6*(s**2))
            hy = (-dy) * (6*s - 6*(s**2))

            bx = p * (-nx) * (2*s - 6*(s**2) + 4*(s**3))
            by = p * (-ny) * (2*s - 6*(s**2) + 4*(s**3))

            vx = (hx + bx) / T_leg
            vy = (hy + by) / T_leg

        return [vx, vy]

    sol = solve_ivp(rhs, (0, T), [x0, y0], t_eval=t_eval, rtol=1e-9, atol=1e-12)
    x = sol.y[0]
    y = sol.y[1]

    d = np.sqrt((x - xc)**2 + (y - yc)**2)
    min_dist = np.min(d)

    # check clearance
    if min_dist >= R:
        if best_p is None or abs(p) < abs(best_p):
            best_p = p
            best_min_dist = min_dist

if best_p is None:
    print("No p in the search range cleared the obstacle. Increase the p range.")
else:
    print("Chosen p =", best_p)
    print("Minimum distance to obstacle =", best_min_dist)
    print("Obstacle radius R =", R)

    # rerun once with best_p for plotting
    p = best_p

    def rhs(t, X):
        if t <= T_leg:
            tl = t
            s = tl / T_leg
            hx = dx * (6*s - 6*(s**2))
            hy = dy * (6*s - 6*(s**2))
            bx = p * nx * (2*s - 6*(s**2) + 4*(s**3))
            by = p * ny * (2*s - 6*(s**2) + 4*(s**3))
            vx = (hx + bx) / T_leg
            vy = (hy + by) / T_leg
        else:
            tl = t - T_leg
            s = tl / T_leg
            hx = (-dx) * (6*s - 6*(s**2))
            hy = (-dy) * (6*s - 6*(s**2))
            bx = p * (-nx) * (2*s - 6*(s**2) + 4*(s**3))
            by = p * (-ny) * (2*s - 6*(s**2) + 4*(s**3))
            vx = (hx + bx) / T_leg
            vy = (hy + by) / T_leg
        return [vx, vy]

    sol = solve_ivp(rhs, (0, T), [x0, y0], t_eval=t_eval, rtol=1e-9, atol=1e-12)
    x = sol.y[0]
    y = sol.y[1]

    plt.figure()
    plt.plot(x, y, label=f"trajectory (p={p})")
    plt.scatter([x0, xT, x0], [y0, yT, y0], color="black")
    plt.text(x0, y0, "  A")
    plt.text(xT, yT, "  B")

    # draw obstacle circle
    theta = np.linspace(0, 2*np.pi, 300)
    ox = xc + R*np.cos(theta)
    oy = yc + R*np.sin(theta)
    plt.plot(ox, oy, linewidth=2, label="obstacle")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.title("Obstacle avoidance with circular obstacle")
    plt.legend()
    plt.show()

    # Error comparison of numerical vs predicted
    '''err = np.max(np.sqrt((x_num - x_pred)**2 + (y_num - y_pred)**2))
    print(f"p = {p}, max parametric error = {err:.2e}")'''