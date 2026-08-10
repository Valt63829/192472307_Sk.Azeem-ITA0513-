import cv2
import numpy as np

# Load image
img = cv2.imread('./Input/ex14.jpg')

# Source and destination points (4 pairs)
pts_src = np.float32([[100,100],[400,100],[400,400],[100,400]])
pts_dst = np.float32([[80,120],[420,80],[380,420],[120,380]])

# Compute Homography matrix
H, status = cv2.findHomography(pts_src, pts_dst)

# Apply transformation
warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

# Show results
cv2.imshow('Original', img)
cv2.imshow('Homography Transform', warped)
cv2.waitKey(0); cv2.destroyAllWindows()

# Save output
cv2.imwrite('homography_transformed.jpg', warped)
