import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_energy(aI, rI, iL, thetaI, pthetaI):
    """Calculate orbital energy Ee using the given parameters."""
    A = (2*aI**2*rI + aI**2*rI**2 + rI**4 + 
         aI**2*(aI**2 + (-2 + rI)*rI)*np.cos(thetaI)**2)
    B = -4*aI*iL*rI
    C = (-aI**2*pthetaI**2 + 2*iL**2*rI + 2*pthetaI**2*rI - aI**2*rI**2 - 
         iL**2*rI**2 - pthetaI**2*rI**2 + 2*rI**3 - rI**4 - 
         iL**2*(aI**2 + (-2 + rI)*rI)*(1/np.tan(thetaI))**2)
    
    discriminant = B**2 - 4*A*C
    Ee = (-B + np.sqrt(discriminant)) / (2*A)
    return Ee

def calculate_carter_constant(aI, Ee, iL, thetaI, pthetaI):
    """Calculate the Carter constant Q."""
    Ce = (pthetaI**2 + np.cos(thetaI)**2 * 
          (aI**2*(1 - Ee**2) + iL**2/(np.sin(thetaI)**2)))
    return Ce

def equations(tau, y, a, L, Ee, Ce):
    """Differential equations for the Kerr geodesics."""
    r, phi, theta, pr, ptheta = y
    
    # Common denominators and terms
    denom1 = a**2 * np.cos(theta)**2 + r**2
    denom2 = a**2 - 2*r + r**2
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cot_theta = 1/np.tan(theta)
    csc_theta = 1/sin_theta
    
    # Derivatives
    drdtau = (pr * denom2) / denom1
    
    # dprdtau calculation
    term1 = a**4 * (L - a*Ee)**2 * cos_theta**2
    term2 = a**4 * (L**2 * cos_theta**2 * cot_theta**2 + ptheta**2) * r
    term3 = a**2 * (-a**2 * Ee**2 + 2*a*Ee*L - L**2 + 
                  2*a*Ee*(a*Ee + L)*cos_theta**2 - 
                  4*L**2*cot_theta**2 - 4*ptheta**2)*r**2
    term4 = (4*a**2*Ee**2 - 8*a*Ee*L + 4*L**2 - 
             4*a**2*Ee**2*cos_theta**2 + 
             4*L**2*cot_theta**2 + 2*a**2*L**2*cot_theta**2 + 
             2*(2 + a**2)*ptheta**2)*r**3
    term5 = (-2*a**2*Ee**2 + 6*a*Ee*L - 4*L**2 + 
             a**2*Ee**2*cos_theta**2 - 
             4*L**2*cot_theta**2 - 4*ptheta**2)*r**4
    term6 = (L**2*csc_theta**2 + ptheta**2)*r**5
    term7 = -Ee**2 * r**6
    term8 = pr**2 * denom2**2 * (a**2*cos_theta**2 - r**2 + a**2*r*sin_theta**2)
    
    numerator_pr = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
    dprdtau = numerator_pr / (denom1**2 * denom2**2)
    
    # dphidtau
    dphidtau = (a**2 * L * cot_theta**2 + 
                2*(a*Ee - L - L*cot_theta**2)*r + 
                L*csc_theta**2 * r**2) / (denom1 * denom2)
    
    # dthetadtau
    dthetadtau = ptheta / denom1
    
    # dpthetadtau
    bracket_term = ((Ce + a**2*(-1 + Ee**2)*cos_theta**2 - L**2*cot_theta**2)*denom2 - 
                   (Ce + (L - a*Ee)**2 + r**2)*denom2 + 
                   (a*L - Ee*(a**2 + r**2))**2)
    
    term1_ptheta = (2*a**2*cos_theta*bracket_term*sin_theta)/denom2
    term2_ptheta = -a**2*ptheta**2*np.sin(2*theta)
    term3_ptheta = -a**2*pr**2*denom2*np.sin(2*theta)
    term4_ptheta = denom1*(2*L**2*cot_theta + 2*L**2*cot_theta**3 - 
                          a**2*(-1 + Ee**2)*np.sin(2*theta))
    
    numerator_ptheta = term1_ptheta + term2_ptheta + term3_ptheta + term4_ptheta
    dpthetadtau = numerator_ptheta / (2*denom1**2)
    
    return [drdtau, dphidtau, dthetadtau, dprdtau, dpthetadtau]

# Preset values
preset_values = {
    'constant_radius': {
        'pT': 1200, 'aI': 0.99, 'rI': 4, 'iL': 2, 
        'thetaI': np.pi/3, 'pthetaI': 0.767851, 'frame': 4.5, 'tailLength': 1200, 'zoomManual': False
    },
    'closed_orbit': {
        'pT': 300, 'aI': 0.9, 'rI': 4, 'iL': 2.148, 
        'thetaI': 1.037, 'pthetaI': 0, 'frame': 4.2, 'tailLength': 350, 'zoomManual': False
    },
    'spiral_capture': {
        'pT': 150, 'aI': 0.0, 'rI': 10, 'iL': 3.5, 
        'thetaI': np.pi/2, 'pthetaI': 0, 'frame': 4.5, 'tailLength': 350, 'zoomManual': False
    },
    'unstable_circular_capture': {
        'pT': 150, 'aI': 0.0, 'rI': 4, 'iL': 3.99999, 
        'thetaI': np.pi/2, 'pthetaI': 0, 'frame': 4.5, 'tailLength': 350, 'zoomManual': False
    },
    'unstable_circular_escape': {
        'pT': 100, 'aI': 0.0, 'rI': 4, 'iL': 4.00001, 
        'thetaI': np.pi/2, 'pthetaI': 0, 'frame': 4.5, 'tailLength': 350, 'zoomManual': False
    },
    'equatorial_zoom_whirl': {
        'pT': 330, 'aI': 0.99, 'rI': 25, 'iL': 2.427, 
        'thetaI': np.pi/2, 'pthetaI': 0, 'frame': 25, 'tailLength': 330, 'zoomManual': False
    },
    'orbit_reverse_capture': {
        'pT': 150, 'aI': 0.9, 'rI': 4, 'iL': -4.5, 
        'thetaI': np.pi/2, 'pthetaI': 0, 'frame': 4.2, 'tailLength': 350, 'zoomManual': False
    },
    '3d_zoom_whirl': {
        'pT': 150, 'aI': 0.99, 'rI': 10, 'iL': 1.05769, 
        'thetaI': np.pi/2, 'pthetaI': 2.89, 'frame': 4, 'tailLength': 150, 'zoomManual': True
    }
}

def plot_orbit(preset_name):
    """Plot the orbit based on the selected preset."""
    params = preset_values[preset_name]
    
    # Calculate energy and Carter constant
    Ee = calculate_energy(params['aI'], params['rI'], params['iL'], 
                         params['thetaI'], params['pthetaI'])
    Ce = calculate_carter_constant(params['aI'], Ee, params['iL'], 
                                  params['thetaI'], params['pthetaI'])
    
    # Initial conditions
    y0 = [params['rI'], 0, params['thetaI'], 0, params['pthetaI']]
    
    # Event for when particle reaches horizon
    def event(tau, y, *args):
        return y[0] - (1 + np.sqrt(1 - params['aI']**2)) * 1.02
    event.terminal = True
    
    # Solve the ODE
    sol = solve_ivp(equations, [0, params['pT']], y0, 
                    args=(params['aI'], params['iL'], Ee, Ce), 
                    events=event, dense_output=True, rtol=1e-8, atol=1e-8)
    
    # Determine if particle plunged
    hole_size = 1 + np.sqrt(1 - params['aI']**2)
    planet_has_plunged = len(sol.t_events[0]) > 0 and abs(sol.y_events[0][-1,0] - hole_size) <= 0.05*hole_size
    
    # Prepare for plotting
    tau_vals = np.linspace(max(0, sol.t[-1] - params['tailLength']), sol.t[-1], 1000)
    solution = sol.sol(tau_vals)
    
    # Convert to Cartesian coordinates
    x = solution[0] * np.sin(solution[2]) * np.cos(solution[1])
    y = solution[0] * np.sin(solution[2]) * np.sin(solution[1])
    z = solution[0] * np.cos(solution[2])
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot orbit
    ax.plot(x, y, z, label='Orbit', linewidth=1)
    
    # Plot black hole
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = hole_size * np.outer(np.cos(u), np.sin(v))
    ys = hole_size * np.outer(np.sin(u), np.sin(v))
    zs = hole_size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='black', alpha=0.7)
    
    # Plot ergosphere
    outer_ergo = 2  # Outer boundary of ergosphere
    xs_ergo = outer_ergo * np.outer(np.cos(u), np.sin(v))
    ys_ergo = outer_ergo * np.outer(np.sin(u), np.sin(v))
    zs_ergo = hole_size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs_ergo, ys_ergo, zs_ergo, color='gray', alpha=0.2)
    
    # Plot planet if it hasn't plunged
    if not planet_has_plunged:
        planet_size = 0.02 * params['frame']
        planet_pos = [x[-1], y[-1], z[-1]]
        ax.scatter(*planet_pos, color='green', s=100*planet_size)
    
    # Set view and limits
    ax.set_xlim([-params['frame'], params['frame']])
    ax.set_ylim([-params['frame'], params['frame']])
    ax.set_zlim([-params['frame'], params['frame']])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add text annotations
    ax.text2D(0.05, 0.95, f"Energy = {Ee:.4f}", transform=ax.transAxes)
    ax.text2D(0.05, 0.90, f"Carter Q = {Ce:.4f}", transform=ax.transAxes)
    ax.text2D(0.05, 0.85, f"Preset: {preset_name.replace('_', ' ')}", transform=ax.transAxes)
    
    plt.tight_layout()
    plt.show()

# Plot all available orbits
for preset_name in preset_values.keys():
    plot_orbit(preset_name)