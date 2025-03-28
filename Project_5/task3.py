"""
Leaksmy Heng
March 26, 2025
CS5330
Project5: Recognition using Deep Networks - Task 3
"""

import torch
import matplotlib.pyplot as plt
import torchvision

from Project_5.constants import Constants
from Project_5.buildingNetworks import Net


# Define transformations for Greek dataset
class GreekTransform:
    def __init__(self, train=True):
        self.train = train
        self.resize_size = (128, 128)
        self.train_resize_size = (28, 28)

    def __call__(self, x):
        # if test then resize
        if not self.train:
            x = torchvision.transforms.Resize(size=self.resize_size)(x)
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        # Invert for easier identification
        return torchvision.transforms.functional.invert(x)


def transfer_learning_on_greek_letter(model_path: str):
    """
    Function to modify the model's result that we train with the numbers.
    """
    # (1) generate the MNIST network (you should import your code from task 1)
    network = Net()
    # (2) read an existing model from a file and load the pre-trained weights
    network.load_state_dict(torch.load(model_path))
    # (3) freeze the network weights
    for layers in network.parameters():
        layers.requires_grad = False
    # (4) replace the last layer with a new Linear layer with three nodes
    network.fc2 = torch.nn.Linear(50, 3)
    # printout and fc2 does show output feature of 3
    print('network', network)
    return network


def load_greek_data_set(training_data_path: str, testing_data_path: str, batch_size_train: int = 5, batch_size_test: int = 2):
    """
    Load greek data set from the specified training data path and testing data path.
    """
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_data_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size_train,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            testing_data_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # set train to false for test data
                GreekTransform(train=False),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, test_loader


def test(network, test_loader, test_losses):
    """
    Function to test the network.
    """
    # set neural network model to evaluation mode as prior to this we set it to training mode
    network.eval()
    test_loss = 0
    correct = 0

    # disables gradient computation as we are not training it
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # Compute the negative log likelihood loss
            test_loss += torch.nn.functional.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy = correct / len(test_loader.dataset)
    print(f'\nTest set: Avg. loss: {test_loss}, Accuracy: {accuracy} ({100. * correct / len(test_loader.dataset)}%)\n')
    return accuracy


def train_model(train_loader, test_loader, network, epochs, optimizer):
    """
    Function to train the model.
    """
    training_errors = []
    testing_errors = []
    test_losses = []
    accuracies = []

    total_training_data = 0
    correct_training_data = 0

    # looping through each epoch
    for epoch in range(1, epochs + 1):
        network.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # does not have to specify whether we are using cpu or gpu
            # because i only have CPU in my machine and by default pytorch uses CPU

            # reset gradient to 0 on each batch
            optimizer.zero_grad()
            # compute network output using forward pass
            output = network(data)
            # Negative log likelihood loss
            loss = torch.nn.functional.nll_loss(output, target)
            # get the loss using backward propagation
            loss.backward()
            # update model param
            optimizer.step()

            # get output of model's predictions for a batch of inputs
            _, pred = torch.max(output.data, 1)
            # Keep track of the total number of training samples processed.
            total_training_data += target.size(0)
            # Keep track of the total number of correctly predicted training samples
            correct_training_data += (pred == target).sum().item()

            training_error = correct_training_data / total_training_data
            training_errors.append(training_error)

            test_accuracy = test(network, test_loader, test_losses)
            accuracies.append(test_accuracy)
            testing_errors.append(1 - test_accuracy)

    return training_errors, testing_errors


def main():
    """
    Main function to execute task3.
    """
    network = transfer_learning_on_greek_letter('model.pth')
    # load data
    testing_path = r"./Image/greek_test"
    training_path = r"./Image/greek_train"
    train_loader, test_loader = load_greek_data_set(training_data_path=training_path, testing_data_path=testing_path)

    # initialized optimizer
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=Constants.LEARNING_RATE,
                                momentum=Constants.MOMENTUM
                                )
    # Train the model and collect errors/accuracies
    training_errors, testing_errors = train_model(train_loader=train_loader,
                                                  test_loader=test_loader,
                                                  network=network,
                                                  optimizer=optimizer,
                                                  epochs=Constants.N_EPOCHS
                                                  )

    # plot the data
    plt.plot(range(1, len(training_errors) + 1), training_errors)
    plt.title("Training Errors Chart")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Training Errors")
    plt.show()

    plt.plot(range(1, len(testing_errors) + 1), testing_errors)
    plt.title("Testing Errors Chart")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Testing Errors")
    plt.show()


if __name__ == '__main__':
    main()
