import cv2

img = cv2.imread("ex8.jpg")

big = cv2.resize(img, None, fx=2, fy=2)
small = cv2.resize(img, None, fx=0.5, fy=0.5)

cv2.imshow("Original", img)
cv2.imshow("Bigger", big)
cv2.imshow("Smaller", small)
while True:
    if cv2.waitKey(0) & 0xFF == ord('1'):
      break
cv2.destroyAllWindows()