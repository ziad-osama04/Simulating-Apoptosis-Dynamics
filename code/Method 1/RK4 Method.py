import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Define parameters from Table 3.3
params = {
    "ahif": 1.52,
    "ao2": 1.8,
    "ap53": 0.05,
    "a3": 0.9,
    "a4": 0.2,
    "a5": 0.001,
    "a7": 0.7,
    "a8": 0.06,
    "a9": 0.1,
    "a10": 0.7,
    "a11": 0.2,
    "a12": 0.1,
    "a13": 0.1,
    "a14": 0.05
}

# System of ODEs (Eqs. 3.1 to 3.6)
def dydt(t, y, p):
    yhif, yo2, yp300, yp53, ycasp, ykp = y
    dy = np.zeros(6)

    dy[0] = p['ahif'] - p['a3']*yo2*yhif - p['a4']*yhif*yp300 - p['a7']*yp53*yhif
    dy[1] = p['ao2'] - p['a3']*yo2*yhif + p['a4']*yhif*yp300 - p['a11']*yo2
    dy[2] = -p['a4']*yhif*yp300 - p['a5']*yp300*yp53 + p['a8']
    dy[3] = p['ap53'] - p['a5']*yp53*yp300 - p['a9']*yp53
    dy[4] = p['a9']*yp53 + p['a12'] - p['a13']*ycasp
    dy[5] = -p['a10']*ycasp*ykp + p['a11']*yo2 - p['a14']*ykp

    return dy

# Runge-Kutta 4th order method
def runge_kutta_4(f, y0, t0, tf, dt, p):
    t_vals = np.arange(t0, tf + dt, dt)
    y_vals = np.zeros((len(t_vals), len(y0)))
    y = y0.copy()

    for i, t in enumerate(t_vals):
        y_vals[i] = y
        k1 = dt * f(t, y, p)
        k2 = dt * f(t + dt/2, y + k1/2, p)
        k3 = dt * f(t + dt/2, y + k2/2, p)
        k4 = dt * f(t + dt, y + k3, p)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_vals, y_vals

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Define settings and initial conditions FIRST ---
    t0, tf, dt = 0, 100, 0.01
    
    # Define BOTH initial conditions here
    y0_1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y0_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # --- Now, solve for both ---
    print("Running RK4 solver for standard IC...")
    t_vals_1, y_vals_1 = runge_kutta_4(dydt, y0_1, t0, tf, dt, params)
    
    print("Running RK4 solver for zero IC...")
    t_vals_0, y_vals_0 = runge_kutta_4(dydt, y0_0, t0, tf, dt, params)
    
    print("Solvers finished. Generating plots...")
    
    # --- Export to CSV
    df1 = pd.DataFrame(y_vals_1, columns=["yhif", "yo2", "yp300", "yp53", "ycasp", "ykp"])
    df1["time"] = t_vals_1
    df1.to_csv("apoptosis_rk4_output_y0_1.csv", index=False)
    
    df0 = pd.DataFrame(y_vals_0, columns=["yhif", "yo2", "yp300", "yp53", "ycasp", "ykp"])
    df0["time"] = t_vals_0
    df0.to_csv("apoptosis_rk4_output_y0_0.csv", index=False)
    
    # --- Plotting the Results ---
    labels = ["y_hif(t)", "y_o2(t)", "y_p300(t)", "y_p53(t)", "y_casp(t)", "y_kp(t)"]
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle('Apoptosis Model: RK4 Solution for Two Initial Conditions', fontsize=16)

    for i, ax in enumerate(axs.flat):
        ax.plot(t_vals_1, y_vals_1[:, i], 'b-', label='IC = [1,0,...]')
        ax.plot(t_vals_0, y_vals_0[:, i], 'r--', label='IC = [0,0,...]')
        ax.set_title(f'{labels[i]}')
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.grid(True)
        ax.legend()

    plt.show()
