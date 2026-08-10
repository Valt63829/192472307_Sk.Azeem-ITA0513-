import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("./Input/ex22.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Laplacian mask with positive center coefficient
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# Apply Laplacian filter
laplacian = cv2.filter2D(img, cv2.CV_64F, kernel)

# Sharpen the image
sharp = img.astype(np.float64) - laplacian
sharp = np.clip(sharp, 0, 255).astype(np.uint8)

# Display
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sharp)
plt.title("Sharpened Image")
plt.axis("off")

plt.show()