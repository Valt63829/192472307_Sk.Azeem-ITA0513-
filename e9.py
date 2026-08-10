import cv2

# Read the image
image = cv2.imread("./Input/ex9.jpg")

# Rotate 90° Clockwise
clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Rotate 90° Counter-Clockwise
counter_clockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Display the images
cv2.imshow("Original Image", image)
cv2.imshow("90 Degree Clockwise", clockwise)
cv2.imshow("90 Degree Counter Clockwise", counter_clockwise)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()