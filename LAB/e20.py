import cv2
import numpy as np

# Read image
img = cv2.imread('./Input/ex20.jpg')

# Sharpening kernel (Laplacian with negative center, folded into one kernel)
kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

sharpened = cv2.filter2D(img, -1, kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Sharpened Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./Output/sharpened.jpg', sharpened)