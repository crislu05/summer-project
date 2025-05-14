import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt

# Set default renderer for Plotly
pio.renderers.default = 'browser'

# ----------------------------
# Constants
# ----------------------------
G = 6.67430e-11       # Gravitational constant [m^3 kg^-1 s^-2]
c = 3e8               # Speed of light [m/s]
AU = 1.496e11         # Astronomical Unit [m]
M_bh = 4.3e6 * 1.989e30  # Mass of the black hole [kg]
a_spin = 0.9 * G * M_bh / c**2  # Spin parameter for Kerr black hole

# Masses of the stars
m1 = 10 * 1.989e30    # Mass of S1 [kg]
m2 = 15 * 1.989e30    # Mass of S2 [kg]

# ----------------------------
# Orbital Parameters
# ----------------------------
# S1 parameters
P1 = 94.1 * 365.25 * 24 * 3600  # Orbital period [s]
e1 = 0.358                      # Eccentricity

# S2 parameters
P2 = 16.05 * 365.25 * 24 * 3600  # Orbital period [s]
e2 = 0.8847                      # Eccentricity

# Semi-major axes using Kepler's third law
a1 = ((G * M_bh * P1**2) / (4 * np.pi**2))**(1/3)
a2 = ((G * M_bh * P2**2) / (4 * np.pi**2))**(1/3)

# Pericenter distances
r_peri1 = a1 * (1 - e1)
r_peri2 = a2 * (1 - e2)

# Speeds at pericenter
v_peri1 = np.sqrt(G * M_bh * (1 + e1) / r_peri1)
v_peri2 = np.sqrt(G * M_bh * (1 + e2) / r_peri2)



# S1 initial conditions at pericenter
x1, y1, z1 = (r_peri1, 0, 0)
vx1, vy1, vz1 = (0, v_peri1, 0)

# S2 initial conditions at pericenter
x2, y2, z2 = (r_peri2, 0, 0)
vx2, vy2, vz2 = (0, v_peri2, 0)

# Combine initial conditions
initial_conditions = [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]

# ----------------------------
# Differential Equations
# ----------------------------
def derivatives(t, y):
    # Unpack positions and velocities
    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2 = y
    r1 = np.array([x1, y1, z1])
    v1 = np.array([vx1, vy1, vz1])
    r2 = np.array([x2, y2, z2])
    v2 = np.array([vx2, vy2, vz2])

    # Distances
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r12 = r2 - r1
    r12_mag = np.linalg.norm(r12)

    # Gravitational acceleration from black hole
    a1_bh = -G * M_bh * r1 / r1_mag**3
    a2_bh = -G * M_bh * r2 / r2_mag**3

    # Schwarzschild correction
    schwarz1 = (3 * G * M_bh) / (r1_mag * c**2) * a1_bh
    schwarz2 = (3 * G * M_bh) / (r2_mag * c**2) * a2_bh

    # Kerr correction
    L1 = np.cross(r1, v1)
    L2 = np.cross(r2, v2)
    kerr1 = (2 * G * a_spin / (c**2 * r1_mag**3)) * np.cross(L1, r1)
    kerr2 = (2 * G * a_spin / (c**2 * r2_mag**3)) * np.cross(L2, r2)

    # Mutual gravitational acceleration
    a1_mutual = G * m2 * r12 / r12_mag**3
    a2_mutual = -G * m1 * r12 / r12_mag**3

    # Total accelerations
    a1_total = a1_bh + schwarz1 + kerr1 + a1_mutual
    a2_total = a2_bh + schwarz2 + kerr2 + a2_mutual

    # Return derivatives
    return [vx1, vy1, vz1, *a1_total, vx2, vy2, vz2, *a2_total]

# ----------------------------
# Time Integration
# ----------------------------
# Total simulation time: 2 periods of S1
T_total = 2 * P1
t_span = (0, T_total)
t_eval = np.linspace(*t_span, 5000)

# Solve the ODE
sol = solve_ivp(derivatives, t_span, initial_conditions, t_eval=t_eval)

# Extract positions for plotting
x1_sol, y1_sol, z1_sol = sol.y[0]/AU, sol.y[1]/AU, sol.y[2]/AU
x2_sol, y2_sol, z2_sol = sol.y[6]/AU, sol.y[7]/AU, sol.y[8]/AU

# ----------------------------
# Animation Frames
# ----------------------------
frames = []
for i in range(2, len(x1_sol), 2):
    frames.append(go.Frame(data=[
    go.Scatter3d(x=x1_sol[:i], y=y1_sol[:i], z=z1_sol[:i], mode='lines', line=dict(color='blue')),
    go.Scatter3d(x=x2_sol[:i], y=y2_sol[:i], z=z2_sol[:i], mode='lines', line=dict(color='green')),
    go.Scatter3d(x=[x1_sol[i-1]], y=[y1_sol[i-1]], z=[z1_sol[i-1]], mode='markers', marker=dict(size=5, color='red'), name='S1'),
    go.Scatter3d(x=[x2_sol[i-1]], y=[y2_sol[i-1]], z=[z2_sol[i-1]], mode='markers', marker=dict(size=5, color='yellow'), name='S2'),
    go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=6, color='black'), name='Sgr A*')
]))


# ----------------------------
# Layout and Figure
# ----------------------------
layout = go.Layout(
    title='S1 and S2 Orbiting a Kerr Black Hole',
    scene=dict(
        xaxis_title='x [AU]',
        yaxis_title='y [AU]',
        zaxis_title='z [AU]',
        aspectmode='data'
    ),
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        buttons=[
            dict(label='Play', method='animate', args=[None, {"frame": {"duration": 50, "redraw": True},
                                                              "fromcurrent": True, "mode": "immediate"}]),
            dict(label='Pause', method='animate', args=[[None], {"mode": "immediate", "frame": {"duration": 0}}])
        ]
    )]
)

fig = go.Figure(
    data=[
        go.Scatter3d(x=[x1_sol[0]], y=[y1_sol[0]], z=[z1_sol[0]],
                     mode='lines', line=dict(color='blue', width=3)),
        go.Scatter3d(x=[x1_sol[0]], y=[y1_sol[0]], z=[z1_sol[0]],
                     mode='markers', marker=dict(size=5, color='red')),
        go.Scatter3d(x=[x2_sol[0]], y=[y2_sol[0]], z=[z2_sol[0]],
                     mode='lines', line=dict(color='green', width=3)),
        go.Scatter3d(x=[x2_sol[0]], y=[y2_sol[0]], z=[z2_sol[0]],
                     mode='markers', marker=dict(size=5, color='yellow')),
        go.Scatter3d(x=[0], y=[0], z=[0],
                     mode='markers', marker=dict(size=6, color='black'), name='Sgr A*')
    ],
    layout=layout,
    frames=frames
)
fig.show()


x1, y1, z1 = sol.y[0], sol.y[1], sol.y[2]
vx1, vy1, vz1 = sol.y[3], sol.y[4], sol.y[5]
x2, y2, z2 = sol.y[6], sol.y[7], sol.y[8]
vx2, vy2, vz2 = sol.y[9], sol.y[10], sol.y[11]

# Vectors and magnitudes
r1 = np.vstack([x1, y1, z1]).T
v1 = np.vstack([vx1, vy1, vz1]).T
r2 = np.vstack([x2, y2, z2]).T
v2 = np.vstack([vx2, vy2, vz2]).T
r1_mag = np.linalg.norm(r1, axis=1)
r2_mag = np.linalg.norm(r2, axis=1)
v1_mag = np.linalg.norm(v1, axis=1)
v2_mag = np.linalg.norm(v2, axis=1)
r12_mag = np.linalg.norm(r2 - r1, axis=1)

# Energies
KE1 = 0.5 * m1 * v1_mag**2
KE2 = 0.5 * m2 * v2_mag**2
PE1 = -G * M_bh * m1 / r1_mag
PE2 = -G * M_bh * m2 / r2_mag
PE_mutual = -G * m1 * m2 / r12_mag

# Angular momentum
L1 = np.cross(r1, m1 * v1)
L2 = np.cross(r2, m2 * v2)
L1_mag = np.linalg.norm(L1, axis=1)
L2_mag = np.linalg.norm(L2, axis=1)
L_total = L1 + L2
L_total_mag = np.linalg.norm(L_total, axis=1)

# Total energy
E_total = KE1 + KE2 + PE1 + PE2 + PE_mutual

# Time axis in years
t_years = sol.t / (365.25 * 24 * 3600)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axs[0].plot(t_years, KE1, label='KE S1')
axs[0].plot(t_years, KE2, label='KE S2')
axs[0].plot(t_years, PE1, label='PE S1')
axs[0].plot(t_years, PE2, label='PE S2')
axs[0].plot(t_years, PE_mutual, label='Mutual PE')
axs[0].set_ylabel("Energy (J)")
axs[0].legend()
axs[0].set_title("Kinetic and Potential Energies")

axs[1].plot(t_years, L1_mag, label='|L| S1')
axs[1].plot(t_years, L2_mag, label='|L| S2')
axs[1].plot(t_years, L_total_mag, label='|L Total|', linestyle='--')
axs[1].set_ylabel("Angular Momentum (kg·m²/s)")
axs[1].legend()
axs[1].set_title("Angular Momentum")

axs[2].plot(t_years, E_total, label='Total Energy', color='purple')
axs[2].set_ylabel("Energy (J)")
axs[2].set_xlabel("Time (years)")
axs[2].legend()
axs[2].set_title("Total Mechanical Energy of the System")

plt.tight_layout()
plt.show()