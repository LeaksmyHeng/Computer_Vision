"""
Leaksmy Heng
March 22, 2025
CS5330
Project5: Recognition using Deep Networks
"""

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
        """
        Function to create a forward propagation for the network.
        """
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


def train(epoch: int, network: Net, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim, train_losses: list, train_counter: list):
    """
    Function used to train the next work in each epoch and recorded it to the train_losses and train_counter.

    :param epoch: the epoch number (the number of epoch is specified in constants.py)
    :param network: network that we used to train the model on
    :param train_loader: the training dataset
    :param optimizer: the optimizer that we used when during the backward propagation to calculate the weight. We are using batch stocastic gradient descent.
    :param train_losses: the list that we use to record the train loss data
    :param train_counter: the list that we use to record the train counter
    """
    # Set the module in training mode
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # resets the gradients of all optimized parameters to zero
        # this is because this is train in different epoch so each epoch should be started with 0
        optimizer.zero_grad()
        # Takes the input data and passes it through all the layers of the network
        # the output of our network
        output = network(data)
        # compute a negative log-likelihodd loss between the output and the ground truth label
        loss = F.nll_loss(output, target)
        # using backward propagation to get the loss and use the optimizer step
        loss.backward()
        optimizer.step()

        # log the information. Doing the mod to avoid way to noisy log
        if batch_idx % Constants.LOG_INTERVAL == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

            # Task 1D - Save the network to a file
            # save the state dictionaries of both the neural network model and the optimizer
            torch.save(network.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')


def test(network, test_loader, test_losses):
    """Function to test the network.

    :param network: your deep network
    :param test_loader: test dataset
    """
    # set neural network model to evaluation mode as prior to this we set it to training mode
    network.eval()
    test_loss = 0
    correct = 0

    # disables gradient computation as we are not training it
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # Compute the negative log likelihood loss.
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Avg. loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n')


def train_network():
    """
    Function to train the network that we have written above.
    """
    # initialized the network
    network = Net()
    # optimizer the network using stochastic gradient descent
    optimizer = optim.SGD(network.parameters(),
                          lr=Constants.LEARNING_RATE,
                          momentum=Constants.MOMENTUM
                          )

    # get training data
    train_loader = Project().get_mnist_training_dataset()
    test_loader = Project().get_mnist_test_dataset()

    # keeping track of training losses and test losses
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(Constants.N_EPOCHS + 1)]

    # one epoch at a time
    test(network, test_loader, test_losses)
    for epoch in range(1, Constants.N_EPOCHS + 1):
        # Train the network
        train(epoch, network, train_loader, optimizer, train_losses, train_counter)
        # test the network
        test(network, test_loader, test_losses)

    # Plot the training and testing accuracy in a graph

    logger.info('Train counter is: ', train_counter)
    logger.info('Train loss is:', train_losses)
    logger.info('Tet counter is:', test_counter)
    logger.info(test_losses)
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show(block=True)


def main():
    """Main function used to train the network."""
    logger.info('Start training the data')
    train_network()
    logger.info('Finish training data')


if __name__ == "__main__":
    main()
