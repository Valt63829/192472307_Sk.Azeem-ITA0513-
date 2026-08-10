import cv2
import numpy as np

img = cv2.imread('./Input/ex15.jpg')
pts_src = np.float32([[100,100],[400,100],[400,400],[100,400]])
pts_dst = np.float32([[80,120],[420,80],[380,420],[120,380]])

H, _ = cv2.findHomography(pts_src, pts_dst)
warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

cv2.imshow('Original', img)
cv2.imshow('DLT Transform', warped)
cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('transformed_dlt.jpg', warped)
