import cv2

img1 = cv2.imread('./Input/ex27.jpg')

img2 = cv2.imread('./Input/ex26.jpg')

crop = img2[50:150, 50:150]

x, y = 200, 200   

img1[y:y+crop.shape[0], x:x+crop.shape[1]] = crop

cv2.imshow("Original Background", cv2.imread('./Input/ex27.jpg'))
cv2.imshow("Inserted Image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./Output/ex27_image_pasted.jpg", img1)
