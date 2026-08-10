import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./Input/ex17.jpg", 0)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")
plt.axis("off")
plt.show()