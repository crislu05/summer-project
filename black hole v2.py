import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'browser'

# ----------------------------
# Constants
# ----------------------------
G = 6.67430e-11           # Gravitational constant [m^3 kg^-1 s^-2]
c = 3e8                   # Speed of light [m/s]
M = 8.2e36                # Mass of Sgr A* [kg]
AU = 1.496e11             # 1 Astronomical Unit [m]

# Spin parameter a = J / (M * c), near-maximal spin
a_spin = 0.9 * G * M / c**2   # [m]

# ----------------------------
# Initial conditions
# ----------------------------
e = 0.88
r0 = 1.8e13                              # Pericentre distance [m]
v0 = np.sqrt(G * M * (1 + e) / r0)       # Tangential speed at pericentre [m/s]

# Give it z offset and tilt to break symmetry
initial_conditions = [
    r0, 0, 0.2 * r0,                     # x, y, z
    0, v0, 0.1 * v0                      # vx, vy, vz
]

# ----------------------------
# Differential equation system
# ----------------------------
def derivatives(t, y):
    x, y_, z, vx, vy, vz = y
    r_vec = np.array([x, y_, z])
    v_vec = np.array([vx, vy, vz])
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Newtonian gravity
    a_grav = -G * M * r_vec / r**3

    # Schwarzschild correction (perihelion precession)
    schwarz_factor = (3 * G * M) / (r * c**2)
    a_schwarz = schwarz_factor * a_grav

    # Frame dragging (Kerr black hole)
    L_vec = np.cross(r_vec, v_vec)                             # Orbital angular momentum
    a_kerr = (2 * G * a_spin / (c**2 * r**3)) * np.cross(L_vec, r_vec)

    # Total acceleration
    a_total = a_grav + a_schwarz + a_kerr
    return [vx, vy, vz, *a_total]

# ----------------------------
# Time integration
# ----------------------------
T_orbit = 5.06e8                            # Period of S2 orbit ~16 years [s]
t_span = (0, 10 * T_orbit)                  # Simulate 10 orbits
t_eval = np.linspace(*t_span, 15000)        # Evaluation points

sol = solve_ivp(derivatives, t_span, initial_conditions, t_eval=t_eval)

# ----------------------------
# Extract and convert to AU
# ----------------------------
x = sol.y[0] / AU
y = sol.y[1] / AU
z = sol.y[2] / AU

# ----------------------------
# Plot 3D interactive orbit
# ----------------------------
orbit = go.Scatter3d(
    x=x, y=y, z=z,
    mode='lines',
    line=dict(color='blue'),
    name='S2 Orbit'
)

black_hole = go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=6, color='black'),
    name='Sgr A* (Black Hole)'
)

layout = go.Layout(
    title='S2 Orbit around a Kerr Black Hole (Frame Dragging Included)',
    scene=dict(
        xaxis_title='x [AU]',
        yaxis_title='y [AU]',
        zaxis_title='z [AU]',
        aspectmode='data'
    ),
    legend=dict(x=0.05, y=0.95)
)

fig = go.Figure(data=[orbit, black_hole], layout=layout)
fig.show()
