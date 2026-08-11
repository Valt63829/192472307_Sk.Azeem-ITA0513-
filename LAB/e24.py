import cv2
import numpy as np

# Load grayscale image
img = cv2.imread('./Input/ex24.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur (low-pass filter)
blurred = cv2.GaussianBlur(img, (5,5), 0)

# High-boost filtering (A > 1)
A = 1.7
high_boost = cv2.addWeighted(img, A, blurred, -1, 0)

cv2.imshow("Original", img)
cv2.imshow("High-Boost Sharpened", high_boost)
cv2.imwrite('./Output/high_boost_sharpened.jpg', high_boost)
cv2.waitKey(0)
cv2.destroyAllWindows()
