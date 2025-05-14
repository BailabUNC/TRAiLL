import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cmap import Colormap
from scipy.ndimage import zoom  # Add this import

# Load the augmented tensor data
file_path = 'data/augmented/augmented_dataset_letter_200.pt'
augmented_data = torch.load(file_path)
print(augmented_data['features'].shape)  # Check the shape of the loaded data

N, A, T, C = augmented_data['features'].shape

random.seed(0)
n = random.randint(0, N - 1)
a = random.randint(0, A - 1)

sample = augmented_data['features'][n, a]
sample = sample.numpy()
print(sample.shape)  # Check the shape of the sample

# set up an interval so that we can select ten frames from the sample
interval = T // 10
# Create a list of ten frames and interpolate each by 5x
frames = [zoom(sample[i * interval, :].reshape(6, 8), zoom=5, order=3) for i in range(10)]

# Prepare grid for surface plot (interpolated size)
x = np.linspace(0, 7, 8 * 5)
y = np.linspace(0, 5, 6 * 5)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the aspect ratio to make the z-axis visually taller
ax.set_box_aspect([1, 1, 2])  # [x, y, z] ratio; increase z for more height

# Plot each frame as a surface, stacked along the z-axis
for idx, Z in enumerate(frames):
    ax.plot_surface(X, Y, Z + idx * 5, alpha=0.7, cmap=Colormap('cmocean:balance').to_mpl())

# Remove grid, text, and axis
ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()