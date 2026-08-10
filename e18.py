import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./Input/ex18.jpg", 0)

sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)

plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")
plt.show()