import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G    = 6.67430e-11   # m^3 kg^-1 s^-2
c    = 3.0e8         # m/s
M    = 8.2e36        # kg (mass of Sgr A*)
e    = 0.88          # eccentricity of S2's orbit
r0   = 1.8e13        # m (pericentre distance)
AU   = 1.496e11      # m

# Initial tangential speed at pericentre
v0 = np.sqrt(G * M * (1 + e) / r0)

# Initial state [x, y, vx, vy] at pericentre (y=0, moving in +y)
initial_conditions = [r0, 0.0, 0.0, v0]

# Time span: simulate e.g. 10 full orbits
T_orbit = 5.06e8         # seconds per S2 orbit ≈ 16 years
t_span = (0.0, 10*T_orbit)
t_eval = np.linspace(t_span[0], t_span[1], 20000)

def derivatives(t, y):
    x, y_, vx, vy = y
    r_vec   = np.array([x, y_])
    v_vec   = np.array([vx, vy])
    
    r       = np.linalg.norm(r_vec)
    v2      = np.dot(v_vec, v_vec)
    r_dot_v = np.dot(r_vec, v_vec)
    
    # Newtonian acceleration
    a_N = -G * M / r**3 * r_vec
    
    # 1PN correction (vector form)
    a_1PN = (G * M) / (c**2 * r**3) * (
        (4 * G * M / r - v2) * r_vec
        + 4 * r_dot_v * v_vec
    )
    
    a_total = a_N + a_1PN
    return [vx, vy, a_total[0], a_total[1]]

# Integrate
sol = solve_ivp(derivatives, t_span, initial_conditions,
                t_eval=t_eval, rtol=1e-9, atol=1e-12)

# Extract solution
x = sol.y[0] / AU
y = sol.y[1] / AU

# Plot
plt.figure(figsize=(8,8))
plt.plot(0, 0, 'ko', label='Sgr A*')
plt.plot(x, y, '-', lw=1, label='S2 orbit (with 1PN)')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Relativistic Orbit of S2 around Sagittarius A*')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
