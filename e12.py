import cv2
import numpy as np

img = cv2.imread('./Input/ex12.jpg')
pts_src = np.float32([[100,100],[400,100],[400,400],[100,400]])
pts_dst = np.float32([[80,120],[420,80],[380,420],[120,380]])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

cv2.imshow('Original', img)
cv2.imshow('Warped', warped)
cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('perspective_transformed.jpg', warped)
