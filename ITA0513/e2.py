import cv2

image = cv2.imread("ex3.jpg")

blurred = cv2.blur(image, (15, 15))

cv2.imwrite("blurred.jpg", blurred)