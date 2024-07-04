import time
import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the model
model = mujoco.MjModel.from_xml_path('hand.xml')
data = mujoco.MjData(model)

# Get the index of the end-effector site
site_name = "end_effector_site"
site_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

# Linear trajectory parameters
X_start = np.array([0.0, 0.0, 5.0])  # Starting point of the trajectory
X_end = np.array([0.04, 0.04, 5.0])  # End point of the trajectory (4 cm cross)
v_max = 0.4  # Maximum velocity (m/s)
T = (15 / 8) * np.linalg.norm(X_end - X_start) / v_max  # Duration to achieve maximum velocity at the middle

# Polynomial coefficients for s(t)
a0, a1, a2 = 0, 0, 0
a3 = 10 / T**3
a4 = -15 / T**4
a5 = 6 / T**5

# Time steps
dt = 0.01
time_steps = np.arange(0, T + dt, dt)

# Storage for results
results = []

def s(t):
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

def ds(t):
    return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4

def dds(t):
    return 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

# Inverse kinematics residual function
def residual(q, x_desired):
    data.qpos[:] = q
    mujoco.mj_fwdPosition(model, data)
    x_current = data.site_xpos[site_index]
    return np.linalg.norm(x_current - x_desired)

# Main loop to generate and record the trajectory
for t in time_steps:
    s_t = s(t)
    ds_t = ds(t)
    dds_t = dds(t)

    X_t = X_start + s_t * (X_end - X_start)
    dX_t = ds_t * (X_end - X_start)[:, np.newaxis]
    ddX_t = dds_t * (X_end - X_start)

    # Solve inverse kinematics to find joint angles
    q0 = data.qpos.copy()  # Initial guess for the optimization
    res = minimize(residual, q0, args=(X_t,), bounds=[(-np.pi, np.pi)]*model.nq)
    q_t = res.x

    # Compute Jacobian and its derivative
    mujoco.mj_fwdPosition(model, data)
    J = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, J, None, site_index)
    J_pinv = np.linalg.pinv(J)

    dq_t = J_pinv @ dX_t
    dJ_dt = (J - J_prev) / dt if t > 0 else np.zeros_like(J)
    
    ddX_t = ddX_t.reshape((3, 1))
    print(f"dJ_dt: {dJ_dt.shape}")
    print(f"dX_t: {dX_t.shape}")
    print(f"J_pinv: {J_pinv.shape}")
    print(f"ddX_t: {ddX_t.shape}")
    print(f"dq_t: {dq_t.shape}")
    print(f"J: {J.shape}")
    print(f"J: {J.shape}")
    print(f"dof: {model.nv}")
    # ddq_t = J_pinv @ (ddX_t - dJ_dt @ dX_t)
    ddq_t = np.ndarray((4, 1))

    # Set state in the data object
    data.qpos[:] = q_t
    # data.qvel[:] = dq_t.reshape((4, 1))
    data.qacc[:] = ddq_t.reshape((4, ))

    # Compute inverse dynamics to get joint torques
    mujoco.mj_inverse(model, data)
    tau_t = data.qfrc_inverse.copy()

    # Record the results
    results.append({
        'time': t,
        'qpos': q_t.copy(),
        'qvel': dq_t.copy(),
        'qacc': ddq_t.copy(),
        'tau': tau_t.copy(),
        'X': X_t.copy(),
        'dX': dX_t.copy(),
        'ddX': ddX_t.copy()
    })

    # Store the previous Jacobian
    J_prev = J.copy()

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv('robot_trajectory_data.csv', index=False)

# Plot the results
fig, axs = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

for i in range(model.nq):
    axs[0].plot(df['time'], df['qpos'].apply(lambda x: x[i]), label=f'Joint {i+1}')
    axs[1].plot(df['time'], df['qvel'].apply(lambda x: x[i]), label=f'Joint {i+1}')
    axs[2].plot(df['time'], df['qacc'].apply(lambda x: x[i]), label=f'Joint {i+1}')
    axs[3].plot(df['time'], df['tau'].apply(lambda x: x[i]), label=f'Joint {i+1}')

axs[0].set_ylabel('Position (rad)')
axs[1].set_ylabel('Velocity (rad/s)')
axs[2].set_ylabel('Acceleration (rad/s^2)')
axs[3].set_ylabel('Torque (Nm)')
axs[3].set_xlabel('Time (s)')

for ax in axs:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('robot_trajectory_plots.png')
plt.close()

# Record a video of the trajectory using MuJoCo viewer
viewer = mujoco.MjViewer(model)
for t in time_steps:
    s_t = s(t)
    X_t = X_start + s_t * (X_end - X_start)
    q_t = df.loc[df['time'] == t, 'qpos'].values[0]
    data.qpos[:] = q_t
    mujoco.mj_step(model, data)
    viewer.render()

# Save the video
viewer.capture_video('robot_trajectory_video.mp4')
viewer.close()
