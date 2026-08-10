import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./Input/ex23.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Stronger blur
blur = cv2.GaussianBlur(img, (9,9), 3)

# Stronger sharpening
sharp = cv2.addWeighted(img, 2.0, blur, -1.0, 0)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sharp)
plt.title("Unsharp Masking")
plt.axis("off")

plt.show()