import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import neuron, layer, learning
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from torch.utils.tensorboard import SummaryWriter
import os
import argparse


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


def main():
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-data-dir', default='C:/Users/maher/Dataset', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')

    args = parser.parse_args()
    print(args)

    net = parametric_lif_net.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    print(net)

    net.to(args.device)

    # Load the trained model
    checkpoint_path = 'C:/Users/maher/Documents/KURF/Spike Jelly/logs/T16_b16_adam_lr0.001_c128_amp_cupy/checkpoint_latest.pth'
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    net.load_state_dict(checkpoint['net'])

    # Set the network to evaluation mode
    net.eval()

    # Load your preprocessed data
    data_path = 'C:/Users/maher/Documents/KURF/Test/maher_282_np.npz'
    data = np.load(data_path)

    # Extract the frames data using the correct key
    frames = data['frames']

    # Convert the data to Torch tensors and adjust dimensions
    frames_tensor = torch.from_numpy(frames).unsqueeze(0).float().to(args.device)

    # Pass the preprocessed data through the network
    predictions = net(frames_tensor)

    # Perform post-processing on the predictions if needed
    # Assuming the predictions are in the shape [batch_size, num_classes]
    # You can obtain the classified label by taking the argmax along the class dimension
    labels_pred = predictions.argmax(dim=1)

    # Print the classified labels
    print(labels_pred)


if __name__ == '__main__':
    main()

    

