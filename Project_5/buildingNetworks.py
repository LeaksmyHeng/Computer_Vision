"""
Leaksmy Heng
March 22, 2025
CS5330
Project5: Recognition using Deep Networks
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.optim as optim
import matplotlib.pyplot as plt

from Project_5.gettingData import Project
from Project_5.constants import Constants

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """Function to train the network.

    Reference:
    https://nextjournal.com/gkoehler/pytorch-mnist
    """

    def __init__(self):
        super(Net, self).__init__()

        # kernel size is 5x4 for both conv layers
        # conv1: 1 input channel (grayscale), outputs 10 channels (features) - 10 filters, 5x5 kernel size
        # conv2: 10 channels, outputs 20 channels (deeper features) - 20 filters, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=Constants.KERNEL_SIZE)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=Constants.KERNEL_SIZE)

        # drop out rate with the default of 0.5
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # dimensionality reduction
        self.fc1 = nn.Linear(320, 50)
        # final classes
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Using ReLu as the non-linear activation function
        # Max pooling with 2x2 window after each conv layer
        # Conv Block 1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Conv Block 2
        # Conv → Dropout → Pool → ReLU
        # 2x2 max pooling + ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Converts 3D tensors to 1D vectors
        # 50 nodes with ReLU
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # Final log_softmax for classification
        return F.log_softmax(x)





if __name__ == "__main__":
    main(sys.argv)
