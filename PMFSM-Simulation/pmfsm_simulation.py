# ==============================================================================
# PMFSM (Permanent Magnet Flux Switching Machine)
# Cascade Speed and Position Control Simulation (Field-Oriented Control)
# Academic Framework for Graduate-Level Electrical Machine Modeling
# Compatible with Python / JupyterLab / Anaconda (Spyder)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# SIMULATION TIMING PARAMETERS
# ==============================================================================
dt = 0.0001          # Sampling time step (seconds)
tmax = 4.0           # Total simulation duration (seconds)
time = np.arange(0, tmax, dt)

# ==============================================================================
# MACHINE PHYSICAL PARAMETERS (Synchronized with LaTeX Report Data)
# ==============================================================================
Rs = 0.5              # Stator phase resistance (Ohm)
Ld = 0.008            # d-axis inductance (H) - 8 mH
Lq = 0.01             # q-axis inductance (H) - 10 mH
lambda_m = 0.15       # Permanent magnet flux linkage (Wb)
p = 4                 # Number of pole pairs
J = 0.02              # Rotor and load total inertia (kg*m^2)
B = 0.001             # Viscous friction coefficient (N*m*s/rad)
Vdc = 300             # DC bus voltage limit (V)
TL_m = 2.0            # Nominal magnitude of load torque (Nm)

# ==============================================================================
# CONTROL SYSTEM PARAMETERS (Optimized Graduate-Level PI Gains)
# ==============================================================================
# Position Loop PI Tuning (Configured to eliminate steady-state error)
Kp_theta = 25.0
Ki_theta = 2.0

# Speed Loop PI Tuning (Balanced for fast response and smooth damping)
Kp_w = 1.5
Ki_w = 20.0

# Inner d-q Current Loop PI Controllers (High bandwidth to track electrical dynamics)
Kp_id = 20.0
Ki_id = 400.0

Kp_iq = 20.0
Ki_iq = 400.0

# Operational Saturation Limits
wmax = 300            # Maximum allowable angular speed (rad/s)
Tmax = 50             # Maximum allowable electromagnetic torque (Nm)
imax = 50             # Maximum allowable stator current (A)

# ==============================================================================
# INITIAL CONDITIONS & PI INTEGRAL ACCUMULATORS
# ==============================================================================
id_val = 0.0
iq_val = 0.0
wm = 0.0              # Initial mechanical speed (rad/s)
theta = 0.0           # Initial rotor position (rad)

# PI Controller Integral States (Accumulators for numerical integration)
theta_error_int = 0.0
werror_int = 0.0
id_error_int = 0.0
iq_error_int = 0.0

# ==============================================================================
# DATA STORAGE LOGS FOR POST-PROCESSING
# ==============================================================================
theta_list = []
theta_ref_list = []
speed_list = []
speed_ref_list = []
id_list = []
iq_list = []
iq_ref_list = []
Te_list = []
power_mech_list = []
power_elec_list = []

# ==============================================================================
# DYNAMIC SIMULATION LOOP (Forward Euler Numerical Integration)
# ==============================================================================
for t in time:

    # --------------------------------------------------------------------------
    # POSITION REFERENCE PROFILE (90 degrees for 0-2s, resets to 0 degrees)
    # --------------------------------------------------------------------------
    theta_deg_ref = 90.0
    if t > 2.0:
        theta_deg_ref = 0.0

    theta_ref = np.deg2rad(theta_deg_ref)

    # --------------------------------------------------------------------------
    # POSITION CONTROLLER (Outer Loop - Positional Form PI)
    # --------------------------------------------------------------------------
    theta_error = theta_ref - theta
    theta_error_int += theta_error * dt
    
    # Calculate speed reference command
    wref = Kp_theta * theta_error + Ki_theta * theta_error_int
    wref = np.clip(wref, -wmax, wmax)

    # --------------------------------------------------------------------------
    # SPEED CONTROLLER (Middle Loop - Positional Form PI)
    # --------------------------------------------------------------------------
    werror = wref - wm
    werror_int += werror * dt
    
    # Calculate electromagnetic torque reference command
    Tref = Kp_w * werror + Ki_w * werror_int
    Tref = np.clip(Tref, -Tmax, Tmax)

    # --------------------------------------------------------------------------
    # CURRENT REFERENCE GENERATION (FOC Strategy: id_ref = 0 for MTPA)
    # --------------------------------------------------------------------------
    id_ref = 0.0
    # Simplified Torque Relation: Te = 1.5 * p * lambda_m * iq (assuming id=0)
    iq_ref = Tref / (1.5 * p * lambda_m)
    iq_ref = np.clip(iq_ref, -imax, imax)

    # --------------------------------------------------------------------------
    # CURRENT CONTROLLERS (Inner Loops - True PI Voltage Command Generation)
    # --------------------------------------------------------------------------
    id_error = id_ref - id_val
    id_error_int += id_error * dt
    Vd = Kp_id * id_error + Ki_id * id_error_int

    iq_error = iq_ref - iq_val
    iq_error_int += iq_error * dt
    Vq = Kp_iq * iq_error + Ki_iq * iq_error_int

    # Inverter Voltage Saturation (DC Bus Overprotection)
    Vd = np.clip(Vd, -Vdc, Vdc)
    Vq = np.clip(Vq, -Vdc, Vdc)

    # --------------------------------------------------------------------------
    # ELECTRICAL SPEED COMPUTATION
    # --------------------------------------------------------------------------
    we = p * wm

    # --------------------------------------------------------------------------
    # ELECTRICAL MODEL (PMFSM State-Space Differential Equations)
    # --------------------------------------------------------------------------
    did_dt = (Vd - Rs * id_val + we * Lq * iq_val) / Ld
    diq_dt = (Vq - Rs * iq_val - we * Ld * id_val - we * lambda_m) / Lq

    # Discrete State Integration
    id_val += did_dt * dt
    iq_val += diq_dt * dt

    # --------------------------------------------------------------------------
    # ELECTROMAGNETIC TORQUE (Including Asymmetric Saliency Effects)
    # --------------------------------------------------------------------------
    Te = 1.5 * p * (lambda_m * iq_val + (Ld - Lq) * id_val * iq_val)

    # --------------------------------------------------------------------------
    # LOAD TORQUE MODEL (Direction-Sensitive Coulomb Friction Representation)
    # --------------------------------------------------------------------------
    if wm > 0.1:
        TL = TL_m
    elif wm < -0.1:
        TL = -TL_m
    else:
        TL = 0.0

    # --------------------------------------------------------------------------
    # MECHANICAL MODEL STATE INTEGRATION
    # --------------------------------------------------------------------------
    dwm_dt = (Te - TL - B * wm) / J
    wm += dwm_dt * dt

    # --------------------------------------------------------------------------
    # ROTOR POSITION UPDATE
    # --------------------------------------------------------------------------
    theta += wm * dt

    # --------------------------------------------------------------------------
    # POWER BALANCE CALCULATIONS
    # --------------------------------------------------------------------------
    power_mech = Te * wm                                 # Mechanical Output Power (W)
    power_elec = 1.5 * (Vd * id_val + Vq * iq_val)       # Electrical Input Power (W)

    # --------------------------------------------------------------------------
    # APPEND REAL-TIME CONVERGED DATA TO LOGS
    # --------------------------------------------------------------------------
    theta_list.append(theta)
    theta_ref_list.append(theta_ref)
    speed_list.append(wm * 30 / np.pi)       # Convert rad/s to RPM
    speed_ref_list.append(wref * 30 / np.pi)
    id_list.append(id_val)
    iq_list.append(iq_val)
    iq_ref_list.append(iq_ref)
    Te_list.append(Te)
    power_mech_list.append(power_mech)
    power_elec_list.append(power_elec)

# ==============================================================================
# PLOTTING AND EXPORT AUTOMATION FOR LATEX SCHEMATICS
# ==============================================================================
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# --- GRAPH 1: SPEED & TORQUE RESPONSES (hiz.png) ---
fig1, ax1 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1[0].plot(time, speed_list, label='Actual Speed ($n$)', color='green', linewidth=1.8)
ax1[0].plot(time, speed_ref_list, '--', label='Reference Speed ($n_{ref}$)', color='orange', linewidth=1.5)
ax1[0].set_ylabel("Speed (RPM)", fontsize=11)
ax1[0].set_title("PMFSM Dynamic Speed Response & Electromagnetic Torque", fontsize=13, fontweight='bold')
ax1[0].legend(loc='upper right')
ax1[0].grid(True, linestyle=':')

ax1[1].plot(time, Te_list, label='Electromagnetic Torque ($T_e$)', color='brown', linewidth=1.8)
ax1[1].axhline(y=TL_m, color='black', linestyle=':', label='Load Torque ($T_L$)')
ax1[1].set_ylabel("Torque (Nm)", fontsize=11)
ax1[1].set_xlabel("Time (Seconds)", fontsize=11)
ax1[1].legend(loc='upper right')
ax1[1].grid(True, linestyle=':')

plt.tight_layout()
plt.savefig('hiz.png', dpi=300)
plt.show()

# --- GRAPH 2: POSITION, CURRENTS & POWER (konum.png) ---
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
ax2[0].plot(time, np.rad2deg(theta_list), label='Actual Position ($\theta$)', color='blue', linewidth=1.8)
ax2[0].plot(time, np.rad2deg(theta_ref_list), '--', label='Reference Position ($\theta_{ref}$)', color='red', linewidth=1.5)
ax2[0].set_ylabel("Position (Degrees)", fontsize=11)
ax2[0].set_title("PMFSM High-Precision Position Tracking & Power Evaluation", fontsize=13, fontweight='bold')
ax2[0].legend(loc='upper right')
ax2[0].grid(True, linestyle=':')

ax2[1].plot(time, iq_list, label='Actual $i_q$ (Torque Component)', color='purple', linewidth=1.8)
ax2[1].plot(time, iq_ref_list, '--', label='Reference $i_{q,ref}$', color='magenta', linewidth=1.2)
ax2[1].plot(time, id_list, label='Actual $i_d$ (Flux Component)', color='cyan', linewidth=1.5)
ax2[1].set_ylabel("Current (Amperes)", fontsize=11)
ax2[1].legend(loc='upper right')
ax2[1].grid(True, linestyle=':')

ax2[2].plot(time, power_elec_list, label='Electrical Input Power ($P_{in}$)', color='red', linewidth=1.2)
ax2[2].plot(time, power_mech_list, label='Mechanical Output Power ($P_{out}$)', color='blue', linewidth=1.8)
ax2[2].set_ylabel("Power (Watts)", fontsize=11)
ax2[2].set_xlabel("Time (Seconds)", fontsize=11)
ax2[2].legend(loc='upper right')
ax2[2].grid(True, linestyle=':')

plt.tight_layout()
plt.savefig('konum.png', dpi=300)
plt.show()

print("[INFO] Simulation completed successfully. 'hiz.png' and 'konum.png' have been saved in high resolution.")