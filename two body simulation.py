import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Physical constants ===
G      = 6.67430e-11    # gravitational constant [m^3 kg^-1 s^-2]
c      = 3.0e8          # speed of light [m/s]
M      = 8.2e36         # mass of Sgr A* [kg]
M_sun  = 1.989e30       # solar mass [kg]
AU     = 1.496e11       # astronomical unit [m]
year   = 3.154e7        # seconds per year
D_pc   = 8266.0         # distance to GC [pc]

# === Stellar masses (assumed) ===
m2 = 14 * M_sun         # mass of S2
m1 = 15 * M_sun         # mass of S1

# === Orbital parameters from Gillessen et al. ===
# S2
e2     = 0.8843        # eccentricity
a2_arc = 0.1251        # semimajor axis [arcsec]
P2_yr  = 16.05         # period [yr]
a2_m   = a2_arc * D_pc * AU
r02    = a2_m * (1 - e2)
v02    = np.sqrt(G*M*(1+e2)/r02)

# S1
e1     = 0.5560
a1_arc = 0.5950
P1_yr  = 166.0
a1_m   = a1_arc * D_pc * AU
r01    = a1_m * (1 - e1)
v01    = np.sqrt(G*M*(1+e1)/r01)

# === Initial conditions at pericentre ===
# [x2, y2, vx2, vy2,  x1, y1, vx1, vy1]
y0 = [
    r02, 0.0,    0.0,  +v02,   # S2 at pericentre, moving +y
    r01, 0.0,    0.0,  +v01    # S1 at pericentre, moving +y
]

# === Time span ===
N2 = 30                        # number of S2 orbits to simulate
t_final = N2 * P2_yr * year
t_eval  = np.linspace(0, t_final, 20000)

def deriv(t, y):
    # unpack state
    x2, y2, vx2, vy2, x1, y1, vx1, vy1 = y

    def accel_bh(x, y_, vx, vy):
        """Acceleration due to central BH with 1PN correction."""
        r_vec   = np.array([x, y_])
        v_vec   = np.array([vx, vy])
        r       = np.linalg.norm(r_vec)
        v2      = np.dot(v_vec, v_vec)
        r_dot_v = np.dot(r_vec, v_vec)
        # Newtonian term
        aN = -G * M / r**3 * r_vec
        # 1PN Schwarzschild correction
        a1PN = (G * M) / (c**2 * r**3) * (
            (4*G*M/r - v2)*r_vec
            + 4*r_dot_v * v_vec
        )
        return aN + a1PN

    # BH accelerations
    a2_bh = accel_bh(x2, y2, vx2, vy2)
    a1_bh = accel_bh(x1, y1, vx1, vy1)

    # mutual Newtonian acceleration between S1 and S2
    r12_vec = np.array([x2 - x1, y2 - y1])
    dist12  = np.linalg.norm(r12_vec) + 1e-20
    # force on S2 by S1
    a2_mutual = -G * m1 / dist12**3 * r12_vec
    # force on S1 by S2 (equal and opposite)
    a1_mutual =  G * m2 / dist12**3 * r12_vec

    # total accelerations
    a2 = a2_bh + a2_mutual
    a1 = a1_bh + a1_mutual

    return [
        vx2, vy2, a2[0], a2[1],
        vx1, vy1, a1[0], a1[1]
    ]

# Integrate ODE
sol = solve_ivp(
    deriv,
    (0, t_final),
    y0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-12
)

# --- After your solve_ivp integration and extracting sol ---
# Compute time in years
t_years = sol.t / year

# Compute radii (m) and speeds (m/s) for each star
r2 = np.sqrt(sol.y[0]**2 + sol.y[1]**2)      # S2 radius
r1 = np.sqrt(sol.y[4]**2 + sol.y[5]**2)      # S1 radius
v2 = np.sqrt(sol.y[2]**2 + sol.y[3]**2)      # S2 speed
v1 = np.sqrt(sol.y[6]**2 + sol.y[7]**2)      # S1 speed

# Compute energies *per unit mass*
U2 = -G * M / r2
U1 = -G * M / r1
KE2 = 0.5 * v2**2
KE1 = 0.5 * v1**2

# Plot potential energy for S2 & S1
plt.figure(figsize=(8, 4))
plt.plot(t_years, U2, label='U (S2)')
plt.plot(t_years, U1, label='U (S1)')
plt.xlabel('Time [years]')
plt.ylabel('Potential Energy per unit mass [J/kg]')
plt.title('Gravitational Potential Energy vs Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot kinetic energy for S2 & S1
plt.figure(figsize=(8, 4))
plt.plot(t_years, KE2, label='KE (S2)')
plt.plot(t_years, KE1, label='KE (S1)')
plt.xlabel('Time [years]')
plt.ylabel('Kinetic Energy per unit mass [J/kg]')
plt.title('Kinetic Energy vs Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# Extract and convert to AU
x2 = sol.y[0] / AU
y2 = sol.y[1] / AU
x1 = sol.y[4] / AU
y1 = sol.y[5] / AU

# Plot
plt.figure(figsize=(8,8))
plt.plot(0, 0, 'ko', label='Sgr A*')
plt.plot(x2, y2, '-', label='S2 Orbit')
plt.plot(x1, y1, '-', label='S1 Orbit')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title(f'Relativistic Orbits of S2 & S1 around Sgr A* ({N2}×S2 Period)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
