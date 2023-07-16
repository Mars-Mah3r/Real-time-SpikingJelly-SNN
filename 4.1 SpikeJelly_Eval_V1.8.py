import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torchvision.transforms import ToTensor
from spikingjelly.activation_based import neuron, layer, learning
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import argparse


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2  # Change the number of input channels to 2
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

    net = DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

    functional.set_step_mode(net, 'm')
    functional.set_backend(net, 'numpy')  # Use the CPU backend instead of CUDA

    print(net)

    net.to(args.device)

    # Load the trained model
    checkpoint_path = 'C:/Users/maher/Documents/KURF/Spike Jelly/logs/T16_b16_adam_lr0.001_c128_amp_cupy/checkpoint_latest.pth'
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    checkpoint_weights = checkpoint['net']

    # Modify the model architecture to match the checkpoint
    for i in range(5):
        checkpoint_weight = checkpoint_weights[f'conv_fc.{i*4}.weight']
        net.conv_fc[i*4].weight.data = checkpoint_weight.squeeze()

    # Set the network to evaluation mode
    net.eval()

    # Load the frame data
    frame_data = np.load('C:/Users/maher/Documents/KURF/Test/maher.npz')['frames']

    # Convert the frame data to float32 and move to device
    frame_data = torch.from_numpy(frame_data).float().unsqueeze(0).to(args.device)

    # Perform inference
    output = net(frame_data)

    print(output)


if __name__ == '__main__':
    main()
