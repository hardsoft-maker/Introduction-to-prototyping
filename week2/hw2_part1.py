import time
import mujoco
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = mujoco.MjModel.from_xml_path('hand.xml')
data = mujoco.MjData(model)

# Joint limits (for each DoF, assumed to be between -pi and pi for simplicity)
num_dofs = model.nq
joint_ranges = [(-np.pi, np.pi)] * num_dofs
num_steps = 10

# Create grid of joint angles
joint_angles = [np.linspace(start, end, num_steps) for start, end in joint_ranges]
all_configs = list(itertools.product(*joint_angles))

# Results storage
results = []

for config in all_configs:
    # Set the joint positions
    data.qpos = np.array(config)
    data.qvel = np.zeros(model.nv)
    data.qacc = np.zeros(model.nv)

    # Calculate inverse dynamics
    mujoco.mj_inverse(model, data)
    
    # Store the results: config and torques
    results.append(config + tuple(data.qfrc_inverse))

# Convert results to a DataFrame
column_names = [f'joint_{i}_angle' for i in range(model.nq)] + [f'joint_{i}_torque' for i in range(model.nq)]
df = pd.DataFrame(results, columns=column_names)

# Save to CSV
df.to_csv('robot_dynamics_results.csv', index=False)

# Plotting the results
torque_columns = [col for col in df.columns if 'torque' in col]
torque_data = df.melt(value_vars=torque_columns, var_name='joint', value_name='torque')

plt.figure(figsize=(12, 6))
sns.violinplot(x='joint', y='torque', data=torque_data)
plt.title('Torque Distribution across Different Joints')
plt.xlabel('Joint')
plt.ylabel('Torque (Nm)')
plt.grid(True)

# Save plot as image
plt.savefig('torque_distribution_violinplot.png')
plt.close()
