import cv2

# Read the image
image = cv2.imread("ex3.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny Edge Detection
edges = cv2.Canny(blurred, 100, 200)

# Save the output image
cv2.imwrite("outline.jpg", edges)

# Display the result
cv2.imshow("Original Image", image)
cv2.imshow("Outline using Canny", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()