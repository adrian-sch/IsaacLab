
import os
import torch
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

script_dir = os.path.dirname(os.path.realpath(__file__))
pickle_path = os.path.join(script_dir, 'sample_inputs.pkl')

with open(pickle_path, 'rb') as f:
    sample_input_real = pickle.load(f)
    
print(sample_input_real)


pickle_path = os.path.join(script_dir, 'obs.pkl')

with open(pickle_path, 'rb') as f:
    sample_input_sim = pickle.load(f)
    
print(sample_input_sim)

# Extract the data
l_lidar_np = sample_input_real['l_lidar_']
lidar_tensor = sample_input_sim['lidar'].cpu().numpy()
goal_tensor = sample_input_sim['goal'].cpu().numpy()


# Get lidar points from distances
lidar_max_range = 5.0

# remove last point to avoid overlap
angles_ccw = np.linspace(np.pi, -np.pi, l_lidar_np.shape[2] + 1)[:-1]

angles_cw = np.linspace(-np.pi, np.pi, l_lidar_np.shape[2] + 1)[:-1]


lidar_points_real = []
lidar_points_sim = [] 
for idx, angle in enumerate(angles_cw):
        # transform polar coordinates to cartesian
        x = l_lidar_np[0, 0, idx] * np.cos(angle) * lidar_max_range
        y = l_lidar_np[0, 0, idx] * np.sin(angle) * lidar_max_range
        lidar_points_real.append((x, y))
        x = lidar_tensor[0, 0, idx] * np.cos(angle) * lidar_max_range
        y = lidar_tensor[0, 0, idx] * np.sin(angle) * lidar_max_range
        lidar_points_sim.append((x, y))


# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)

# Plot the l_lidar_ numpy array as an image
axs[0, 0].imshow(l_lidar_np[0], cmap='viridis', interpolation='nearest', aspect='auto')
axs[0, 0].set_title('Real array')
axs[0, 0].set_xlabel('X-axis')
axs[0, 0].set_ylabel('Y-axis')
fig.colorbar(axs[0, 0].imshow(l_lidar_np[0], cmap='viridis', interpolation='nearest', aspect='auto'), ax=axs[0, 0])

# Plot the lidar tensor as an image
axs[0, 1].imshow(lidar_tensor[0], cmap='viridis', interpolation='nearest', aspect='auto')
axs[0, 1].set_title('Sim tensor')
axs[0, 1].set_xlabel('X-axis')
axs[0, 1].set_ylabel('Y-axis')
fig.colorbar(axs[0, 1].imshow(lidar_tensor[0], cmap='viridis', interpolation='nearest', aspect='auto'), ax=axs[0, 1])

# Plot the l_lidar_ numpy array as a graph
for i in range(l_lidar_np.shape[1]):
    axs[1, 0].plot(l_lidar_np[0, i], label=f'Line {i+1}')
axs[1, 0].set_title('Real (graph)')
axs[1, 0].set_xlabel('Index')
axs[1, 0].set_ylabel('Value')
axs[1, 0].legend()

# Plot the lidar tensor as a graph
for i in range(lidar_tensor.shape[1]):
    axs[1, 1].plot(lidar_tensor[0, i], label=f'Line {i+1}')
axs[1, 1].set_title('Sim (graph)')
axs[1, 1].set_xlabel('Index')
axs[1, 1].set_ylabel('Value')
axs[1, 1].legend()


fig2, axs2 = plt.subplots(1, 2, figsize=(10, 10), dpi=100)

# Plot the lidar points real
lidar_points_real = np.array(lidar_points_real)
colors_real = plt.cm.RdYlGn(np.linspace(0, 1, len(lidar_points_real)))
scatter_real = axs2[0].scatter(lidar_points_real[:, 1], lidar_points_real[:, 0], c=colors_real, label='Real')
axs2[0].set_xlim(-5, 5)
axs2[0].set_ylim(-5, 5)
axs2[0].set_title('Lidar Points Real')
axs2[0].set_xlabel('Y')
axs2[0].set_ylabel('X')
axs2[0].set_aspect('equal')
red_patch = mpatches.Patch(color='red', label='Start (Red)')
green_patch = mpatches.Patch(color='green', label='End (Green)')
axs2[0].legend(handles=[red_patch, green_patch])
axs2[0].grid()

# Plot the lidar points sim
lidar_points_sim = np.array(lidar_points_sim)
colors_sim = plt.cm.RdYlGn(np.linspace(0, 1, len(lidar_points_sim)))
origin = np.zeros((goal_tensor.shape[0], 2))
scatter_sim = axs2[1].scatter(lidar_points_sim[:, 1], lidar_points_sim[:, 0], c=colors_sim, label='Sim')
axs2[1].quiver(origin[0, 1], origin[0, 0], goal_tensor[0, 1], goal_tensor[0, 0], angles='xy', scale_units='xy', scale=1)
axs2[1].set_xlim(-5, 5)
axs2[1].set_ylim(-5, 5)
axs2[1].set_title('Lidar Points Sim')
axs2[1].set_xlabel('Y')
axs2[1].set_ylabel('X')
axs2[1].set_aspect('equal')
axs2[1].legend(handles=[red_patch, green_patch])
axs2[1].grid()

plt.tight_layout()
plt.show()

