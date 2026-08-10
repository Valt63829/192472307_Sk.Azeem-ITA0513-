import cv2
import numpy as np

cap = cv2.VideoCapture('./Input/ex13.mp4')

# Source and destination points
pts_src = np.float32([[100,100],[400,100],[400,400],[100,400]])
pts_dst = np.float32([[80,120],[420,80],[380,420],[120,380]])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    warped = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
    cv2.imshow('Original', frame)
    cv2.imshow('Warped', warped)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
