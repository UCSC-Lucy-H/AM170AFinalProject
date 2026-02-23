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
p_values = [0.0, 1.0, 2.0] # Bulge magnitude


# Bulge/Bump
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

for p in p_values:

# Velocity formula (numerical)
    def rhs(t, X):
        if t <= T_leg:
            tl = t
            s = tl / T_leg
        
            # h'(s) = 6s - 6s^2
            hx = dx*(6*s - 6*(s**2))
            hy = dy*(6*s - 6*(s**2))

            # b'(s) = 2s - 6s^2 + 4s^3
            bx = p*nx*(2*s - 6*(s**2) + 4*(s**3))
            by = p*ny*(2*s - 6*(s**2) + 4*(s**3))

            vx = (hx + bx) / T_leg
            vy = (hy + by) / T_leg

        else:
            tl = t - T_leg
            s = tl / T_leg

            # return direction has signs flipped
            hx = (-dx)*(6*s - 6*(s**2))
            hy = (-dy)*(6*s - 6*(s**2))

            bx = p*(-nx)*(2*s - 6*(s**2) + 4*(s**3))
            by = p*(-ny)*(2*s - 6*(s**2) + 4*(s**3))

            vx = (hx + bx) / T_leg
            vy = (hy + by) / T_leg

        return [vx, vy]

    # Integrate trajectory using solve_ivp
    # Should be using RK45, which is close to ODE45
    # Added tolerances since I noticed the curve would not return
    # to points A or B exactly for large p values
    t_eval = np.linspace(0, T, 1500)
    sol = solve_ivp(rhs, (0, T), [x0, y0], t_eval=t_eval, rtol=1e-9, atol=1e-12)

    x_num = sol.y[0]
    y_num = sol.y[1]


    # Predicted Trajectory
    x_pred = np.zeros_like(t_eval)
    y_pred = np.zeros_like(t_eval)

    for i, t in enumerate(t_eval):
        if t <= T_leg:
            s = t / T_leg
            h = 3*(s**2) - 2*(s**3)
            b = (s**2) * ((1 - s)**2)

            x_pred[i] = x0 + h*dx + p*b*nx
            y_pred[i] = y0 + h*dy + p*b*ny
        else:
            s = (t - T_leg) / T_leg
            h = 3*(s**2) - 2*(s**3)
            b = (s**2) * ((1 - s)**2)

            x_pred[i] = xT + h*(-dx) + p*b*(-nx)
            y_pred[i] = yT + h*(-dy) + p*b*(-ny)

    plt.figure()
    plt.plot(x_num, y_num, label="Numerical (RK45)")
    plt.plot(x_pred, y_pred, "--", label="Predicted")

    plt.scatter([x0, xT, x0], [y0, yT, y0], color="black")
    plt.text(x0, y0, "  A")
    plt.text(xT, yT, "  B")

    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.title(f"Validation Check: predicted vs numerical trajectory (p = {p})")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Error comparison of numerical vs predicted
    err = np.max(np.sqrt((x_num - x_pred)**2 + (y_num - y_pred)**2))
    print(f"p = {p}, max parametric error = {err:.2e}")