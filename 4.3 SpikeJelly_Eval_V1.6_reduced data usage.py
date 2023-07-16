import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.model import parametric_lif_net
import cv2

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    # Load the trained model
    checkpoint_path = 'C:/Users/maher/Documents/KURF/Spike Jelly/logs/T16_b16_adam_lr0.001_c128_amp_cupy/checkpoint_latest.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Define network configuration
    channels = 128  # Update this value based on your trained model
    
    # Create network
    net = parametric_lif_net.DVSGestureNet(channels=channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    net.eval()
    
    # Load preprocessed data
    data_path = 'C:/Users/maher/Documents/KURF/Test/preprocessed/preprocessed_frames.npz'
    data = np.load(data_path)
    frames = data['frames']
    
    # Resize or crop the input data
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (128, 128))  # Resize each frame to 128x128
        resized_frames.append(resized_frame)
    resized_frames = np.array(resized_frames)
    
    # Convert input to tensor and reshape
    time_steps = 8  # Update this value to the desired number of time steps
    frames_tensor = torch.from_numpy(resized_frames).unsqueeze(0).repeat(time_steps, 1, 1, 1, 1).float().to(device)
    
    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.cuda.amp.autocast(enabled=True):
        # Pass the preprocessed data through the network
        predictions = net(frames_tensor)
    
    # Perform post-processing on the predictions if needed
    labels_pred = predictions.argmax(dim=1)
    
    # Print the classified labels
    print(labels_pred)

if __name__ == '__main__':
    main()

