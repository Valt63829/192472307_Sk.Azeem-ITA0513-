import cv2
import numpy as np

# Read image
img = cv2.imread("./Input/ex11.jpg")

# Original and transformed points
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[30, 70], [220, 50], [70, 220]])

# Affine transformation
matrix = cv2.getAffineTransform(pts1, pts2)
result = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

# Display
cv2.imshow("Original", img)
cv2.imshow("Affine Transform", result)
cv2.waitKey(0)
cv2.destroyAllWindows()