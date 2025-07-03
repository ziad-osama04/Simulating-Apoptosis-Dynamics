import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. MODEL AND SOLVER DEFINITIONS
# =============================================================================

# Define parameters
params = {
    "ahif": 1.52, "ao2": 1.8, "ap53": 0.05, "a3": 0.9, "a4": 0.2, "a5": 0.001,
    "a7": 0.7, "a8": 0.06, "a9": 0.1, "a10": 0.7, "a11": 0.2, "a12": 0.1,
    "a13": 0.1, "a14": 0.05
}

# System of ODEs (Corrected version)
def apoptosis_rhs(t, y):
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = y
    dydt = np.zeros_like(y)
    
    # Using keys that exactly match the `params` dictionary
    dydt[0] = params['ahif'] - params['a3']*y_o2*y_hif - params['a4']*y_hif*y_p300 - params['a7']*y_p53*y_hif
    dydt[1] = params['ao2'] - params['a3']*y_o2*y_hif + params['a4']*y_hif*y_p300 - params['a11']*y_o2
    dydt[2] = params['a8'] - params['a4']*y_hif*y_p300 - params['a5']*y_p300*y_p53
    dydt[3] = params['ap53'] - params['a5']*y_p53*y_p300 - params['a9']*y_p53
    dydt[4] = params['a12'] + params['a9']*y_p53 - params['a13']*y_casp
    dydt[5] = -params['a10']*y_casp*y_kp + params['a11']*y_o2 - params['a14']*y_kp
    
    return dydt

# Hand-coded RKF45 integrator
def rkf45_solver(f, t_span, y0, tol=1e-6, h_init=0.1):
    t0, t_end = t_span; t = t0; y = np.array(y0, dtype=float); h = h_init
    ts, ys = [t], [y.copy()]
    b4=np.array([25/216,0,1408/2565,2197/4104,-1/5,0]); b5=np.array([16/135,0,6656/12825,28561/56430,-9/50,2/55])
    k=np.zeros((6,len(y0))); a=[[],[1/4],[3/32,9/32],[1932/2197,-7200/2197,7296/2197],[439/216,-8,3680/513,-845/4104],[-8/27,2,-3544/2565,1859/4104,-11/40]]
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    while t < t_end:
        if t + h > t_end: h = t_end - t
        k[0] = f(t, y)
        for i in range(1, 6): y_temp = y + h * np.dot(a[i], k[:i,:]); k[i] = f(t + c[i]*h, y_temp)
        y4 = y + h * np.dot(b4, k); y5 = y + h * np.dot(b5, k)
        error_norm = np.linalg.norm(y5 - y4, ord=np.inf)
        if error_norm < tol or h < 1e-6:
            t += h; y = y4; ts.append(t); ys.append(y.copy())
        h = min(max(0.9 * h * (tol / (error_norm + 1e-16))**0.25, 0.1 * h), 5.0)
    return np.array(ts), np.array(ys)

# =============================================================================
# 2. MAIN EXECUTION AND PLOTTING
# =============================================================================

if __name__ == "__main__":
    # Define initial conditions and time span
    y0_std = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y0_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_span = (0, 100)

    # Solve for both initial conditions
    print("Running RKF45 solver for both initial conditions...")
    ts_std, ys_std = rkf45_solver(apoptosis_rhs, t_span, y0_std)
    ts_zero, ys_zero = rkf45_solver(apoptosis_rhs, t_span, y0_zero)
    print("Solvers finished. Calculating error and saving individual graphs...")

    # Interpolate for error calculation
    ys_zero_interp = np.array([np.interp(ts_std, ts_zero, ys_zero[:, i]) for i in range(len(y0_std))]).T
    abs_error = np.abs(ys_std - ys_zero_interp)
    
    # Create output directory
    output_dir = "rkf45_individual_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    variable_labels = ["yhif", "yo2", "yp300", "yp53", "ycasp", "ykp"]

    # --- To save individual SOLUTION graphs ---
    for i in range(len(variable_labels)):
        plt.figure(figsize=(8, 6))
        plt.plot(ts_std, ys_std[:, i], 'b-', label='IC = [1,0,...]')
        plt.plot(ts_zero, ys_zero[:, i], 'r--', label='IC = [0,0,...]')
        plt.title(f"RKF45 Solution for {variable_labels[i]}", fontsize=16)
        plt.xlabel('Time', fontsize=12); plt.ylabel('Concentration', fontsize=12)
        plt.grid(True); plt.legend()
        filename = os.path.join(output_dir, f"solution_{variable_labels[i]}.png")
        plt.savefig(filename, dpi=300); plt.close()
        print(f"Saved solution graph: {filename}")

    # --- To save individual ERROR graphs ---
    for i in range(len(variable_labels)):
        plt.figure(figsize=(8, 6))
        plt.plot(ts_std, abs_error[:, i], 'g-')
        plt.title(f"Absolute Difference for {variable_labels[i]}", fontsize=16)
        plt.xlabel('Time', fontsize=12); plt.ylabel('Absolute Difference |y_std - y_zero|', fontsize=12)
        plt.grid(True)
        filename = os.path.join(output_dir, f"error_{variable_labels[i]}.png")
        plt.savefig(filename, dpi=300); plt.close()
        print(f"Saved error graph: {filename}")
        
    print(f"\nAll 12 graphs have been saved successfully in the '{output_dir}' folder.")
