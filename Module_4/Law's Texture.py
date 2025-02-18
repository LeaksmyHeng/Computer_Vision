import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in color (RGB)
image = cv2.imread('pic.0039.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a simple Law's texture filter (for example, a 'level' filter)
level_filter = np.array([[1, 4, 6, 4, 1],
                         [1, 2, 0, -2, -1],
                         [-1, 0, 2, 0, -1],
                         [1, -2, 0, 2, -1],
                         [1, -4, 6, -4, 1]
                         ]
                        )

# Apply the texture filter (convolution)
texture_response = cv2.filter2D(gray_image, -1, level_filter)

average_energy = np.mean(texture_response)
print(f"Average Energy of Gradient Magnitude: {average_energy}")

# Display the filtered image (response of the texture filter)
# plt.imshow(texture_response, cmap='gray')
# plt.title("Law's Texture Filter Response")
# # plt.axis('off')
# plt.show()

plt.figure(figsize=(10, 6))
plt.hist(texture_response.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7)
plt.title("Law's Texture")
plt.xlabel("Gradient Magnitude")
plt.ylabel("Number of Pixels")
plt.show()