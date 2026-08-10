import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./Input/ex19.jpg", 0)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobel_xy = cv2.addWeighted(
    cv2.convertScaleAbs(sobel_x), 0.5,
    cv2.convertScaleAbs(sobel_y), 0.5, 0
)

plt.imshow(sobel_xy, cmap="gray")
plt.title("Sobel XY")
plt.axis("off")
plt.show()