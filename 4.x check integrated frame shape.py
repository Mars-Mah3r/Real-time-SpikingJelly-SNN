import numpy as np
import torch
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

# Load the .npz file
data = np.load('C:/Users/maher/Documents/KURF/Test/maher.npz')
events = {
    't': data['t'],
    'x': data['x'],
    'y': data['y'],
    'p': data['p']
}

# Create an instance of the DVS128Gesture class
dataset = DVS128Gesture('C:/Users/maher/Dataset', train=False)

# Convert events to tensor format
preprocessed_events = {
    't': torch.Tensor(events['t']),
    'x': torch.Tensor(events['x']),
    'y': torch.Tensor(events['y']),
    'p': torch.Tensor(events['p'])
}

# Print the shape of the preprocessed events
print('Preprocessed Events:', preprocessed_events['t'].shape)

# Now you can directly feed preprocessed_events into your trained network for inference or further processing.
