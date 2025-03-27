"""
Leaksmy Heng
March 22, 2025
CS5330
Project5: Recognition using Deep Networks
"""

import logging
import torch
import torchvision
import matplotlib.pyplot as plt

from Project_5.constants import Constants


logger = logging.getLogger(__name__)


class Project:
    def __init__(self):
        self.batch_size_train = Constants.BATCH_SIZE_TRAIN
        self.batch_size_test = Constants.BATCH_SIZE_TEST
        self.random_seed = Constants.RANDOM_SEED
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.random_seed)

    def get_mnist_training_dataset(self, is_shuffle=True):
        """
        Task 1A. Get the MNIST digit data set.
        Function to download the training MNIST dataset.

        Reference:
        https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
        """
        try:
            training_data = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('/files/',
                                           train=True,
                                           download=True,
                                           transform=torchvision.transforms.Compose(
                                               [torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                ])
                                           ),
                batch_size=self.batch_size_train, shuffle=is_shuffle)
            logger.info(f'Length of training data set is {len(training_data)}.')

        except Exception as e:
            raise ValueError(f'Not able to get MNIST training data set with error: {e}')

        return training_data

    def get_mnist_test_dataset(self, is_shuffle=True):
        """
        Task 1A. Get the MNIST digit data set.
        Function to download the test MNIST dataset.

        Reference:
        https://github.com/pytorch/examples/issues/653
        """
        try:

            test_data = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('/files/',
                                           train=False,
                                           download=True,
                                           transform=torchvision.transforms.Compose(
                                               [torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                           )),
                batch_size=self.batch_size_test, shuffle=is_shuffle)
            logger.info(f'Length of test data set is {len(test_data)}.')

        except Exception as e:
            raise ValueError(f'Not able to get MNIST training data set with error: {e}')

        return test_data

    def plot(self):
        """
        Task 1A. Get the MNIST digit data set
        Function to plot the first 6 test dataset. The code is from https://nextjournal.com/gkoehler/pytorch-mnist.

        Reference:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        https://www.linkedin.com/pulse/deep-learning-action-building-training-neural-network-rany-yvuic
        """
        # Grab the data but do not shuffle it as per instruction
        test_loader = self.get_mnist_test_dataset(is_shuffle=False)
        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Label: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show(block=True)
