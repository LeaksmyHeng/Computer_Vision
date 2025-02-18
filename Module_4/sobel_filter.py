import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in color (RGB)
image = cv2.imread('pic.0039.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Sobel filters to compute the gradients
G_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradient
G_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradient

# Compute the gradient magnitude
gradient_magnitude = cv2.magnitude(G_x, G_y)

# computer engery of the gradient
gradient_energy = gradient_magnitude ** 2

average_energy = np.mean(gradient_energy)
print(f"Average Energy of Gradient Magnitude: {average_energy}")


# Plot the image
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(G_x, cmap='gray')
plt.title('Sobel x')
plt.axis('off')

# Gradient Magnitude Image
plt.subplot(1, 4, 3)
plt.imshow(gradient_magnitude, cmap='hot')
plt.title('Gradient Magnitude Sobel')
plt.axis('off')

# Histogram of Gradient Magnitude
plt.subplot(1, 4, 4)
plt.hist(gradient_magnitude.ravel(), bins=50, color='gray', alpha=0.7)
plt.title('Histogram of Gradient Magnitude')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
