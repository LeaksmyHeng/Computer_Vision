"""
Leaksmy Heng
March 26, 2025
CS5330
Project5: Recognition using Deep Networks
"""

import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import models
from torchsummary import summary

from Project_5.buildingNetworks import Net
from Project_5.gettingData import Project


def analyzing_model():
    """
    Task2A: Analyze the first layer. In this task, I get the summary of the model and
    accessing the weight of the first layer in the network then plot it out.
    """
    # initialize the network
    network = Net()
    # load the model or network from model.pth
    network.load_state_dict(torch.load('model.pth'))

    # This is to read the summary of the network
    vgg = models.vgg16()
    summ = summary(vgg, (3, 224, 224))
    print(summ)
    print('\n----------------------------\n')

    # accessing the weight of the first layer
    weight_first_layer = network.conv1.weight
    # the shape shows ten filters, 1 input channel, and each filter is 5x5 in size
    print("Shape of conv1 weights:", weight_first_layer.shape)

    # initialized the pyplot
    plt.figure(figsize=(12, 5))

    # Iterate through the filters and print their weights
    for i in range(weight_first_layer.shape[0]):
        filter_weights = weight_first_layer[i, 0]
        print(f"Filter {i + 1} weights:")
        print(filter_weights)
        print(f"Shape of filter {i + 1}:", filter_weights.shape)

        # Create subplot
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        # Convert filter_weights to numpy array
        filter_np = filter_weights.detach().numpy()
        plt.imshow(filter_np, cmap='viridis')

        # add title
        plt.title(f"\nFilter {i}")
        # not showing axis
        plt.axis('off')

    plt.show()


def applying_filters():
    """
    In this task, I applied the filters that we see on the function above onto to the training set.
    Generate a plot of the 10 filtered images
    """
    # initialize the network
    network = Net()
    # load the model or network from model.pth
    network.load_state_dict(torch.load('model.pth'))

    with torch.no_grad():

        # accessing the weight of the first layer
        weight_first_layer = network.conv1.weight
        # the shape shows ten filters, 1 input channel, and each filter is 5x5 in size
        print("Shape of conv1 weights:", weight_first_layer.shape)

        # Get the training dataset
        train_loader = Project().get_mnist_training_dataset()
        batch = next(iter(train_loader))
        image, batch = batch
        img_list = []

        # Iterate through the filters and print their weights
        for i in range(weight_first_layer.shape[0]):
            img_list.append(weight_first_layer[i, 0].numpy())
            filtered_img = cv2.filter2D(image[0].numpy()[0], -1, weight_first_layer[i, 0].numpy())
            img_list.append(filtered_img)

        # Create a 3x3 grid for 9 examples
        plt.figure(figsize=(9, 6))
        for i in range(20):
            # Create subplot
            plt.subplot(5, 4, i + 1)
            plt.tight_layout()
            plt.imshow(img_list[i], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()



# analyzing_model()
# applying_filters()
