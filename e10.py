import cv2
import numpy as np

img = cv2.imread("ex10.jpg")
M = np.float32([[1, 0, 100], [0, 1, 50]])  # Right 100, Down 50
result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

cv2.imshow("original image", img)
cv2.imshow("Moved Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()