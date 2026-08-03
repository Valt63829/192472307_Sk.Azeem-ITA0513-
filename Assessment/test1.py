import cv2

# Read image
img = cv2.imread("ex8.jpg", 0)

# Global Histogram Equalization
global_eq = cv2.equalizeHist(img)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img)

cv2.imshow("Original", img)
cv2.imshow("Global Equalization", global_eq)
cv2.imshow("CLAHE", clahe_img)

cv2.waitKey(0)
cv2.destroyAllWindows()