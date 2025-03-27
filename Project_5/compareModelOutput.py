"""
Leaksmy Heng
March 26, 2025
CS5330
Project5: Recognition using Deep Networks
"""
import torch
import matplotlib.pyplot as plt

from Project_5.buildingNetworks import Net
from Project_5.gettingData import Project


# get test data and use the non-shuffle version
test_loader = Project().get_mnist_test_dataset(is_shuffle=False)

# Get the first 10 test samples
test_images = [test_loader.dataset[i][0] for i in range(10)]
test_labels = [test_loader.dataset[i][1] for i in range(10)]
test_batch = torch.stack(test_images)

with torch.no_grad():
    # initialized the network
    network = Net()
    # load the saved weights from your trained model
    network.load_state_dict(torch.load('model.pth'))
    network.eval()
    output = network(test_batch)

    # Create a 3x3 grid for 9 examples
    fig = plt.figure(figsize=(12, 5))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()

        # Denormalize the image for proper display
        img = test_batch[i][0] * 0.3081 + 0.1307  # MNIST normalization reversal
        plt.imshow(img, cmap='gray', interpolation='none')

        # Get prediction and ground truth
        pred = output.data.max(1, keepdim=True)[1][i].item()
        true_label = test_labels[i]

        # Show both prediction and actual label
        plt.title(f"Pred: {pred}\nTrue: {true_label}", color='green' if pred == true_label else 'red')
        plt.xticks([])
        plt.yticks([])

    plt.show()
