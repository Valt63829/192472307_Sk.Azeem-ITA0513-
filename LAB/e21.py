import cv2
import numpy as np

# Read image
img = cv2.imread('./Input/ex21.jpg')
img_float = img.astype(np.float64)

# Laplacian mask extended with diagonals (negative center)
laplacian_kernel = np.array([
    [ 1,  1,  1],
    [ 1, -8,  1],
    [ 1,  1,  1]
], dtype=np.float64)

# Apply Laplacian
laplacian = cv2.filter2D(img_float, -1, laplacian_kernel)

# Negative center -> subtract from original to sharpen
sharpened = img_float - laplacian

# Clip to valid range and convert back
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

cv2.imshow('Original Image', img)
cv2.imshow('Sharpened Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./Output/extension_of_diagonals.jpg', sharpened)