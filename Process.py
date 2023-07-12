import numpy as np
import os

# Load the NPZ file
data = np.load('C:/Users/maher/Documents/KURF/Test/maher.npz')

# Extract the data using the correct keys
t = data['t']
x = data['x']
y = data['y']
p = data['p']

# Define the number of frames and the split method
frames_number = 20
split_by = 'number'

# Calculate the number of events per frame
events_per_frame = t.shape[0] // frames_number

# Create the directory for saving the integrated frames
save_dir = 'C:/Users/maher/Documents/KURF/Test/integrated_frames'
os.makedirs(save_dir, exist_ok=True)

# Create the array to store the integrated frames
frames = np.zeros((frames_number, 2, x.max() + 1, y.max() + 1))

# Integrate events into frames
for j in range(frames_number):
    # Calculate the indices for integrating events into the frame
    j_l = events_per_frame * j
    j_r = events_per_frame * (j + 1) if j < frames_number - 1 else t.shape[0]

    # Integrate events into the frame
    for i in range(j_l, j_r):
        frames[j, 0, x[i], y[i]] += 1 if p[i] == 0 else 0
        frames[j, 1, x[i], y[i]] += 1 if p[i] == 1 else 0

    # Save the integrated frame
    save_path = os.path.join(save_dir, f'frame{j}.npz')
    np.savez(save_path, frame=frames[j])

    print(f'Frame {save_path} saved.')

# Print a sample frame
sample_frame = np.load(os.path.join(save_dir, 'frame0.npz'))['frame']
print(sample_frame.shape)
