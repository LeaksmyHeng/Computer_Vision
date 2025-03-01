import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (you can replace it with your own image)
image = cv2.imread('pic.0039.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure the image is in float32 format for precise arithmetic
image_float = image.astype(np.float32)

# Apply two Gaussian blurs with different sigma values (small and large)
sigma_small = 1.0
sigma_large = 3.0
gaussian_small = cv2.GaussianBlur(image_float, (0, 0), sigma_small)
gaussian_large = cv2.GaussianBlur(image_float, (0, 0), sigma_large)

# Compute the Difference of Gaussian (DoG)
dog_image = gaussian_small - gaussian_large

# Amplify high contrast regions in the DoG image
gain_factor = 2.0
enhanced_dog = dog_image * gain_factor

# Add the enhanced DoG to the original image to boost contrast in high-contrast regions
contrast_enhanced_image = image_float + enhanced_dog

# Convert the resulting image back to uint8 for display
contrast_enhanced_image_uint8 = np.clip(contrast_enhanced_image, 0, 255).astype(np.uint8)

# Display the images
plt.figure(figsize=(15, 9))

plt.subplot(1, 5, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 5, 2)
plt.title('Guassian Small')
plt.imshow(gaussian_small, cmap='gray')

plt.subplot(1, 5, 3)
plt.title('Guassian Large')
plt.imshow(gaussian_large, cmap='gray')

plt.subplot(1, 5, 4)
plt.title('Difference of Gaussian (DoG)')
plt.imshow(dog_image, cmap='gray')

plt.subplot(1, 5, 5)
plt.title('Contrast Enhanced Image')
plt.imshow(contrast_enhanced_image_uint8, cmap='gray')

plt.show()
