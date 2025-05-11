import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
M = 8.2e36       # kg (mass of Sgr A*)
e = 0.88         # eccentricity of S2's orbit
r0 = 1.8e13      # m (pericentre distance)

# Initial tangential speed at pericentre
v0 = np.sqrt(G * M * (1 + e) / r0)

# Initial conditions: [x0, y0, vx0, vy0]
initial_conditions = [r0, 0, 0, v0]

# Time span for integration (simulate a full orbit or more)
T_orbit = 10 * 5.06e8
t_span = (0, T_orbit)
t_eval = np.linspace(*t_span, 10000)

# Acceleration function with relativistic correction (optional)
def derivatives(t, y):
    x, y_, vx, vy = y
    r = np.sqrt(x**2 + y_**2)   
    factor = -G * M / r**3 * (1 + (3 * G * M) / (r * (3e8)**2))  # Schwarzschild correction
    ax = factor * x
    ay = factor * y_
    return [vx, vy, ax, ay]

# Solve the system
sol = solve_ivp(derivatives, t_span, initial_conditions, t_eval=t_eval)

# Extract and plot
x, y = sol.y[0], sol.y[1]

AU = 1.496e11  # m

x_au = x / AU
y_au = y / AU

plt.figure(figsize=(8, 8))
plt.plot(0, 0, 'ko', label='Sgr A* (Black Hole)')
plt.plot(x_au, y_au, label='S2 Orbit')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Orbit of S2 around Sagittarius A*')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlim(-2000, 2000)
plt.ylim(-2000, 2000)
plt.show()

