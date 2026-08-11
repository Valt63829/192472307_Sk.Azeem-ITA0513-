import cv2
# Load grayscale image
img = cv2.imread('./Input/ex25.jpg', cv2.IMREAD_GRAYSCALE)

# Compute gradients using Sobel operator
grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude
gradient = cv2.magnitude(grad_x, grad_y)
gradient = cv2.convertScaleAbs(gradient)

# Sharpen by adding gradient
sharpened = cv2.add(img, gradient)

cv2.imshow("Original", img)
cv2.imshow("Gradient Sharpened", sharpened)
cv2.imwrite('./Output/gradient_sharpened.jpg', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
