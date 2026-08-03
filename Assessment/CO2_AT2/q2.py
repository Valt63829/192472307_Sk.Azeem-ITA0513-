import cv2
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread("Industrial_1.jpg")

# Convert from BGR to YCrCb
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Split the channels
Y, Cr, Cb = cv2.split(ycrcb)

# Apply Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
Y_eq = clahe.apply(Y)

# Merge the channels
merged = cv2.merge((Y_eq, Cr, Cb))

# Convert back to BGR
result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

# Convert BGR to RGB for Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(result_rgb)
plt.title("Histogram Equalized")
plt.axis("off")

plt.show()