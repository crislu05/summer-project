import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G      = 6.67430e-11    # m^3 kg^-1 s^-2
c      = 3.0e8          # m/s
M      = 8.2e36         # kg (mass of Sgr A*)
e      = 0.88           # eccentricity of S2's orbit
r0     = 1.8e13         # m (pericentre distance)
AU     = 1.496e11       # m
year   = 3.154e7        # seconds per year
T_orbit = 5.06e8        # seconds per S2 orbit ≈ 16 years

# Initial tangential speed at pericentre
v0 = np.sqrt(G * M * (1 + e) / r0)

# Initial conditions [x, y, vx, vy] at pericentre
initial_conditions = [r0, 0.0, 0.0, v0]

# Time span: simulate 3 orbits for clarity
n_orbits_sim = 10
t_span = (0.0, n_orbits_sim * T_orbit)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

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

# Integrate the orbit
sol = solve_ivp(
    derivatives,
    t_span,
    initial_conditions,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-12
)

# Extract results
t  = sol.t
x  = sol.y[0]
y  = sol.y[1]
vx = sol.y[2]
vy = sol.y[3]

# Compute radius, speed, and energies per unit mass
r  = np.sqrt(x**2 + y**2)
v  = np.sqrt(vx**2 + vy**2)
U  = -G * M / r               # potential energy per unit mass [J/kg]
KE = 0.5 * v**2               # kinetic energy per unit mass [J/kg]

# Convert time to years
t_years = t / year

# Plot Potential Energy vs Time
plt.figure(figsize=(8,4))
plt.plot(t_years, U, color='C1')
plt.xlabel('Time [years]')
plt.ylabel('U [J/kg]')
plt.title('Gravitational Potential Energy per Unit Mass vs Time')
plt.grid(True)
plt.tight_layout()

# Plot Kinetic Energy vs Time
plt.figure(figsize=(8,4))
plt.plot(t_years, KE, color='C2')
plt.xlabel('Time [years]')
plt.ylabel('KE [J/kg]')
plt.title('Kinetic Energy per Unit Mass vs Time')
plt.grid(True)
plt.tight_layout()

plt.show()


# Extract solution and convert to AU
x = sol.y[0] / AU
y = sol.y[1] / AU
t = sol.t

# Determine how many full orbits we actually covered
n_orbits = int(np.floor((t[-1] - t[0]) / T_orbit))

# Prepare a colormap for plotting
colors = plt.cm.viridis(np.linspace(0, 1, n_orbits))

# Plot each orbit separately
plt.figure(figsize=(8, 8))
plt.plot(0, 0, 'ko', label='Sgr A*')

for i in range(n_orbits):
    t_start = i * T_orbit
    t_end   = (i + 1) * T_orbit
    mask = (t >= t_start) & (t < t_end)
    
    plt.plot(
        x[mask], y[mask],
        lw=0.5,                 # thinner line
        color=colors[i],
        label=f'Orbit {i+1}' if i < 5 else None
    )

plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Relativistic Orbit of S2 around Sagittarius A*')
plt.axis('equal')
plt.grid(True)
plt.legend(ncol=2, fontsize='small')
plt.show()


