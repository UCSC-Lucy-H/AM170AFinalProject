# Authors: Matthew Di Fronzo, Kieran Rege, Lucy Han
# Goal: Model curve trajectory for differing bulge magnitudes


# Libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Variables
x0, y0 = 0.0, 0.0 # Starting points
xT, yT = 1.0, 2.0 # Ending points
T = 1.0           # Final time
E_H = 1.0         # Energy for hover
p = 5.0 # Bulge magnitude


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

# Velocity formula
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
t_eval = np.linspace(0, T, 1500)
sol = solve_ivp(rhs, (0, T), [x0, y0], t_eval=t_eval, rtol=1e-9, atol=1e-12)

t = sol.t
x = sol.y[0]
y = sol.y[1]

# Trajectory plot
'''plt.figure()
plt.plot(t, x, label="x(t)")
plt.plot(t, y, label="y(t)")
plt.xlabel("t")
plt.ylabel("Position")
plt.title("x(t) and y(t), p = 0.0")
plt.legend()
plt.grid(True)
plt.show()'''


# Parametric plot (x(t), y(t))
plt.figure()
plt.plot(x, y)
plt.scatter([x[0], x[len(x)//2], x[-1]],
            [y[0], y[len(y)//2], y[-1]])
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.title("Parametric trajectory (x(t), y(t)), p = 5.0")
plt.axis("equal")
plt.grid(True)
plt.show()

# Energy
# Since energy to hover is just 1 and T is 1, our energy use is 1
# Until I put some different variables in there
'''
E_total = E_H * T

print("Total energy used:", E_total)'''