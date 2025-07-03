import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. MODEL AND SOLVER DEFINITIONS
# =============================================================================

# Define parameters
params = {
    'ahif': 1.52, 'ao2': 1.8, 'ap53': 0.05, 'a3': 0.9, 'a4': 0.2,
    'a5': 0.001, 'a7': 0.7, 'a8': 0.06, 'a9': 0.1, 'a10': 0.7,
    'a11': 0.2, 'a12': 0.1, 'a13': 0.1, 'a14': 0.05
}

# System of ODEs
def apoptosis_ode_system(t, y, p):
    """Defines the system of ODEs for apoptosis."""
    yhif, yo2, yp300, yp53, ycasp, ykp = y
    dyhif_dt = p['ahif'] - p['a3']*yo2*yhif - p['a4']*yhif*yp300 - p['a7']*yp53*yhif
    dyo2_dt = p['ao2'] - p['a3']*yo2*yhif + p['a4']*yhif*yp300 - p['a11']*yo2
    dyp300_dt = p['a8'] - p['a4']*yhif*yp300 - p['a5']*yp300*yp53
    dyp53_dt = p['ap53'] - p['a5']*yp300*yp53 - p['a9']*yp53
    dycasp_dt = p['a12'] + p['a9']*yp53 - p['a13']*ycasp
    dykp_dt = -p['a10']*ycasp*ykp + p['a11']*yo2 - p['a14']*ykp
    return np.array([dyhif_dt, dyo2_dt, dyp300_dt, dyp53_dt, dycasp_dt, dykp_dt])

# Euler method solver
def euler_solver(ode_func, y0, t_span, dt, p):
    """A from-scratch implementation of the Explicit Euler method."""
    t_start, t_end = t_span
    t_vals = np.arange(t_start, t_end + dt, dt)
    y_vals = np.zeros((len(t_vals), len(y0)))
    y_vals[0, :] = y0
    for i in range(len(t_vals) - 1):
        y_current = y_vals[i, :]
        slope = ode_func(t_vals[i], y_current, p)
        y_vals[i+1, :] = y_current + dt * slope
    return t_vals, y_vals

# =============================================================================
# 2. MAIN EXECUTION AND PLOTTING
# =============================================================================

if __name__ == "__main__":
    # Define initial conditions and time span
    time_span = (0, 100); time_step = 0.01
    ic_1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ic_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Solve for both initial conditions
    print("Running Euler solver for both initial conditions...")
    t_sol, y_sol1 = euler_solver(apoptosis_ode_system, ic_1, time_span, time_step, params)
    _, y_sol2 = euler_solver(apoptosis_ode_system, ic_2, time_span, time_step, params)
    print("Solvers finished. Calculating error and saving individual graphs...")

    # Calculate Relative Approximate Error
    epsilon = 1e-12
    relative_error = np.abs((y_sol1 - y_sol2) / (y_sol1 + epsilon))
    
    # --- Create a directory to save the graphs ---
    output_dir = "euler_individual_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    variable_labels = ["yhif", "yo2", "yp300", "yp53", "ycasp", "ykp"]

    # --- To save individual SOLUTION graphs ---
    for i in range(y_sol1.shape[1]):
        plt.figure(figsize=(8, 6))
        
        plt.plot(t_sol, y_sol1[:, i], 'b-', label='IC = [1,0,...]')
        plt.plot(t_sol, y_sol2[:, i], 'r--', label='IC = [0,0,...]')
        
        plt.title(f"Euler Method Solution for {variable_labels[i]}", fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Concentration', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        filename = os.path.join(output_dir, f"solution_{variable_labels[i]}.png")
        plt.savefig(filename, dpi=300)
        plt.close() # Close the figure to free memory
        
        print(f"Saved solution graph: {filename}")

    # --- To save individual ERROR graphs ---
    for i in range(relative_error.shape[1]):
        plt.figure(figsize=(8, 6))
        
        plt.plot(t_sol, relative_error[:, i], 'g-')
        
        plt.title(f"Relative Error for {variable_labels[i]}", fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Relative Error |(y1-y2)/y1|', fontsize=12)
        plt.grid(True)
        plt.yscale('log')
        
        filename = os.path.join(output_dir, f"error_{variable_labels[i]}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        
        print(f"Saved error graph: {filename}")
        
    print(f"\nAll 12 graphs have been saved successfully in the '{output_dir}' folder.")
