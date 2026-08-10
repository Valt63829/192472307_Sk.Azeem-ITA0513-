import cv2
import numpy as np

# Read the image in grayscale
image = cv2.imread("ex4.jpg", cv2.IMREAD_GRAYSCALE)

# Create a kernel (structuring element)
kernel = np.ones((5, 5), np.uint8)

# Apply erosion
eroded = cv2.erode(image, kernel, iterations=1)

# Save the output image
cv2.imwrite("eroded_image.jpg", eroded)

# Display the images
cv2.imshow("Original Image", image)
cv2.imshow("Eroded Image", eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()