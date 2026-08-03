import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("bricks.jpg", cv2.IMREAD_GRAYSCALE)

# Sobel 3x3
sobelx3 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely3 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

gradient3 = np.sqrt(sobelx3**2 + sobely3**2)

laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Sobel 5x5
sobelx5 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely5 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

gradient5 = np.sqrt(sobelx5**2 + sobely5**2)

plt.figure(figsize=(15,10))

plt.subplot(2,4,1)
plt.imshow(img,cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2,4,2)
plt.imshow(sobelx3,cmap='gray')
plt.title("Sobel X (3x3)")
plt.axis("off")

plt.subplot(2,4,3)
plt.imshow(sobely3,cmap='gray')
plt.title("Sobel Y (3x3)")
plt.axis("off")

plt.subplot(2,4,4)
plt.imshow(gradient3,cmap='gray')
plt.title("Gradient Magnitude")
plt.axis("off")

plt.subplot(2,4,5)
plt.imshow(laplacian,cmap='gray')
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2,4,6)
plt.imshow(sobelx5,cmap='gray')
plt.title("Sobel X (5x5)")
plt.axis("off")

plt.subplot(2,4,7)
plt.imshow(sobely5,cmap='gray')
plt.title("Sobel Y (5x5)")
plt.axis("off")

plt.subplot(2,4,8)
plt.imshow(gradient5,cmap='gray')
plt.title("Gradient (5x5)")
plt.axis("off")

plt.tight_layout()
plt.show()