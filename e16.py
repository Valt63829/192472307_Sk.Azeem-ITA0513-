import cv2

# Load image in grayscale
img = cv2.imread('./Input/ex16.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(img, 100, 200)

# Show results
cv2.imshow('Original', img)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0); cv2.destroyAllWindows()

# Save output
cv2.imwrite('./Output/canny_edges.jpg', edges)
