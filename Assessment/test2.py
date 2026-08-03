import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Q6.png",0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
mask = np.zeros((rows,cols),np.uint8)

r = 40
cx, cy = cols//2, rows//2
mask[cy-r:cy+r,cx-r:cx+r] = 1

filtered = fshift * mask

img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
img_back = np.abs(img_back)

plt.imshow(img_back,cmap='gray')
plt.show()