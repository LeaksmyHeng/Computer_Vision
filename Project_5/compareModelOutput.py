"""
Leaksmy Heng
March 26, 2025
CS5330
Project5: Recognition using Deep Networks
"""

import glob
import os
import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image

from Project_5.buildingNetworks import Net
from Project_5.gettingData import Project


def compare_model_output(test_batch):
    """
    Function to compare the output between what the model show and what it actually is.
    """
    with torch.no_grad():
        # initialized the network
        network = Net()
        # load the saved weights from your trained model
        network.load_state_dict(torch.load('model.pth'))
        network.eval()
        output = network(test_batch)

        # Create a 3x3 grid for 9 examples
        plt.figure(figsize=(12, 5))
        for i in range(9):
            # Create subplot
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()

            # Denormalize the image for proper display
            img = test_batch[i][0] * 0.3081 + 0.1307  # MNIST normalization reversal
            plt.imshow(img, cmap='gray', interpolation='none')

            # Get prediction and ground truth
            pred = output.data.max(1, keepdim=True)[1][i].item()

            # Show prediction
            plt.title(f"Prediction: {pred}\n")
            plt.xticks([])
            plt.yticks([])

        plt.show()


def compare_my_hand_written_data():
    # Test the network on new inputs
    # read the images, convert them to greyscale, resize them to 28x28 (if necessary)
    # https://stackoverflow.com/questions/57141784/use-relative-path-to-get-all-png-files-in-python
    # https://www.geeksforgeeks.org/python-pil-image-open-method/#
    png_files = glob.glob(
        os.path.join(r'C:\Users\Leaksmy Heng\Documents\GitHub\CS5330\Computer_Vision\Project_5\Image\my_image', "**",
                     "*.png"), recursive=True)

    if png_files:
        # Initialize network once outside the loop
        network = Net()
        network.load_state_dict(torch.load('model.pth'))
        network.eval()

        # Create a single figure for all subplots
        plt.figure(figsize=(12, 5))
        counter = 0

        for image_path in png_files:
            # Stop after 9 images (3x3 grid)

            im = Image.open(image_path)

            # Handle transparency and convert to grayscale
            if im.mode in ('RGBA', 'LA'):
                im = im.convert('L')

            # Resize with better interpolation
            if im.size != (28, 28):
                im = im.resize((28, 28))

            # Apply transformations
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
            tensor = transform(im).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = network(tensor)
                pred = output.argmax(dim=1, keepdim=True).item()

            # Create subplot
            plt.subplot(3, 4, counter + 1)
            plt.tight_layout()

            # Plot image and prediction
            plt.imshow(im, cmap='gray', interpolation='none')
            plt.title(f"Prediction: {pred}\n")
            plt.xticks([])
            plt.yticks([])

            counter += 1

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # # get test data and use the non-shuffle version
    # test_loader = Project().get_mnist_test_dataset(is_shuffle=False)
    # test_images = [test_loader.dataset[i][0] for i in range(10)]
    # test_batch = torch.stack(test_images)
    # compare_model_output(test_batch)

    compare_my_hand_written_data()
