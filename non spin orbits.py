# schwarzschild_orbits.py

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ========================================================================
# 1) Fundamental Schwarzschild quantities (G = c = M = 1 units)
# ========================================================================

def energy_circular(r):
    """
    Return the energy E for a circular geodesic at radius r > 3 in Schwarzschild (M=1, G=c=1).
    Formula: E = sqrt((r - 2)^2 / [r (r - 3)]).
    """
    if r <= 3.0:
        raise ValueError("r must exceed 3 for a circular orbit.")
    return np.sqrt((r - 2)**2 / (r * (r - 3)))

def angular_momentum_circular(r):
    """
    Return the angular momentum L for a circular geodesic at radius r > 3 in Schwarzschild (M=1).
    Formula: L = sqrt(r^2 / (r - 3)).
    """
    if r <= 3.0:
        raise ValueError("r must exceed 3 for a circular orbit.")
    return np.sqrt(r**2 / (r - 3))

def effective_potential(r, L):
    """
    Schwarzschild effective potential for equatorial motion (θ = π/2):
        V_eff(r, L) = (1 - 2/r) * (1 + L^2 / r^2).
    """
    return (1 - 2/r) * (1 + (L**2 / r**2))


# ========================================================================
# 2) Equations of motion in the equatorial plane (θ = π/2 only)
#
# We evolve the three variables [r, φ, p_r], where:
#   p_r ≡ dr/dτ,
#   φ' = L / r^2,
#   p_r' = -½ dV_eff/dr.
#
# (Since θ=π/2 and p_θ=0 throughout, there is no θ‐equation.)
# ========================================================================

def schwarzschild_equatorial_eqns(tau, y, L):
    """
    RHS of the equatorial Schwarzschild geodesic ODEs (M=1 units).  We pass in 'L' as a parameter.
    y = [ r,  φ,  p_r ]
    dr/dτ   = p_r
    dφ/dτ   = L / r^2
    dp_r/dτ = -½ (dV_eff/dr)
    where
      V_eff(r, L) = (1 - 2/r)*(1 + L^2/r^2).
    """
    r, phi, pr = y

    # Compute dV_eff/dr exactly:
    #   V_eff(r) = (1 - 2/r)*(1 + L^2/r^2)
    #   dV_eff/dr =  2/r^2 * (1 + L^2/r^2)
    #              + (1 - 2/r)*(-2 * L^2 / r^3)
    term1     = 2 / r**2 * (1 + (L**2 / r**2))
    term2     = (1 - 2/r) * (-2 * L**2 / r**3)
    dVeff_dr  = term1 + term2

    dr_dtau   = pr
    dphi_dtau = L / (r**2)
    dpr_dtau  = -0.5 * dVeff_dr

    return [dr_dtau, dphi_dtau, dpr_dtau]


# ========================================================================
# 3) A helper to compute the correct p_r(0) from (E, L, r0):
#
#    p_r(0) = ± sqrt[ E^2 - V_eff(r0, L ) ]
#
# We choose the negative root if we want initially inward motion.
# ========================================================================

def pr_from_E(r0, E, L):
    """
    Given a radius r0, energy E, and angular momentum L, return the
    radial momentum p_r(0) = ± sqrt( E^2 - V_eff(r0, L) ).
    Here we choose the negative root (so the particle initially moves inward).
    If E^2 < V_eff, return 0.0 (a turning point).
    """
    V0 = effective_potential(r0, L)
    delta = E**2 - V0
    if delta < 0:
        # At a true turning point (E^2 == V_eff), we want p_r(0) = 0.
        # If numerical noise makes it slightly negative, floor to 0.
        return 0.0
    return -np.sqrt(delta)


# ========================================================================
# 4) Preset initial conditions dictionary
#    Each entry must supply:
#       r0    : initial radius,
#       L     : angular momentum,
#       E     : energy,
#       φ0    : initial φ (we always set φ0=0),
#       pT    : maximum proper‐time of integration,
#       frame : plot‐window half‐width for x,y axes.
#    We will compute p_r(0) below for each case.
# ========================================================================

presets = {
    'circular_stable': {
        'r0':   8.0,
        'L':    angular_momentum_circular(8.0),
        'E':    energy_circular(8.0),
        'φ0':   0.0,
        'pT':   1200,
        'frame': 12
    },
    'circular_unstable': {
        'r0':   5.0,
        'L':    angular_momentum_circular(5.0),
        'E':    energy_circular(5.0),
        'φ0':   0.0,
        'pT':   2500,
        'frame': 8
    },
    'precessing_elliptical': {
        'r0':   17,
        'L':    3.7,
        'E':    np.sqrt(effective_potential(17, 3.7)),
        'φ0':   0,
        'pT':   4000,
        'frame':15
    },
    'zoom_whirl': {
        'r0':   6.5,
        'L':    angular_momentum_circular(4)*(0.9997),
        'E':    np.sqrt(effective_potential(6.5, angular_momentum_circular(4.0)*(0.9997))),
        'φ0':   0.0,
        'pT':   6000,
        'frame':10
    },
    'spiral_fall': {
        'r0':   3.7,
        'L':    4.4,
        'E':    0.1,
        'φ0':   0.0,
        'pT':   500,
        'frame': 5
    },
    'unbound_orbit': {
        'r0':   12.0,
        'L':    5,
        'E':    0.9,
        'φ0':   0.0,
        'pT':   900,
        'frame':20
    }
}


# ========================================================================
# 5) Plotting routine for each preset
# ========================================================================

def plot_schwarzschild_orbit_equatorial(name, params):
    """
    Integrate and plot one Schwarzschild equatorial orbit, given:
       name   : string key for the preset
       params : dictionary with keys ['r0','L','E','φ0','pT','frame']
    """

    r0    = params['r0']
    L     = params['L']
    E     = params['E']
    φ0    = params['φ0']
    pT    = params['pT']
    frame = params['frame']

    # Compute the correct initial p_r(0).  If r0 is a true turning point
    # (i.e. E^2 == V_eff(r0,L)), pr0 will return 0.  Otherwise, it returns
    # the negative root so that the particle moves inward from r0.
    pr0 = pr_from_E(r0, E, L)

    # Initial state vector y0 = [ r(0), φ(0), p_r(0) ]
    y0 = [r0, φ0, pr0]

    # We also stop if r crosses just inside the horizon r=2.
    def event_horizon(tau, y, *args):
        return y[0] - 2.01
    event_horizon.terminal  = True
    event_horizon.direction = -1

    # Integrate using solve_ivp
    sol = solve_ivp(
        fun=lambda τ, y: schwarzschild_equatorial_eqns(τ, y, L),
        t_span=(0, pT),
        y0=y0,
        events=event_horizon,
        dense_output=True,
        rtol=1e-10,
        atol=1e-10
    )

    # Sample many points along the solution for a smooth plot
    τ_vals      = np.linspace(0, sol.t[-1], 2000)
    r_vals, φ_vals = sol.sol(τ_vals)[:2]

    # Convert (r, φ) to Cartesian (x, y) in the equatorial plane; z=0
    x_vals = r_vals * np.cos(φ_vals)
    y_vals = r_vals * np.sin(φ_vals)
    z_vals = np.zeros_like(x_vals)

    # ========== Begin plotting ==========

    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Plot the particle trajectory in orange
    ax.plot(x_vals, y_vals, z_vals, color='orange', lw=2, label='Particle Path')

    # Plot the event horizon as a sphere of radius r = 2
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = 2.0 * np.outer(np.cos(u), np.sin(v))
    ys = 2.0 * np.outer(np.sin(u), np.sin(v))
    zs = 2.0 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='black', alpha=0.7, linewidth=0, zorder=0)

    # Plot ISCO (r=6) and IBCO (r=4) as dashed circles in the equatorial plane
    θ_circ = np.linspace(0, 2*np.pi, 150)
    ax.plot(6.0*np.cos(θ_circ), 6.0*np.sin(θ_circ), 0*θ_circ,
            'b--', lw=1.5, label='ISCO (r=6)')
    ax.plot(4.0*np.cos(θ_circ), 4.0*np.sin(θ_circ), 0*θ_circ,
            'r--', lw=1.5, label='IBCO (r=4)')

    # Set axes limits so the orbit is centered and properly framed
    ax.set_xlim(-frame, frame)
    ax.set_ylim(-frame, frame)
    ax.set_zlim(-frame, frame)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Schwarzschild Orbit — {name.replace('_',' ').title()}")

    # Annotate E, L, and preset name
    ax.text2D(0.04, 0.93, f"Energy  E = {E:.4f}",             transform=ax.transAxes, fontsize=10)
    ax.text2D(0.04, 0.89, f"Angular Momentum  L = {L:.4f}",  transform=ax.transAxes, fontsize=10)
    ax.text2D(0.04, 0.85, f"Preset  = {name}",                transform=ax.transAxes, fontsize=10)
    ax.text2D(0.04, 0.81, "Equatorial Schwarzschild BH (a=0)", transform=ax.transAxes, fontsize=10)

    ax.legend()
    plt.tight_layout()
    plt.show()


# ========================================================================
# 6) Run & plot all presets in sequence
# ========================================================================

if __name__ == "__main__":
    for name, preset in presets.items():
        plot_schwarzschild_orbit_equatorial(name, preset)
