import cv2

# Read the original image
img = cv2.imread("./Input/ex26.jpg")

# Make a copy for watermarking
watermarked = img.copy()

# Watermark text
watermark = "WATERMARK"

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
position = (50, 50)
font_scale = 1
color = (255, 255, 255)
thickness = 2

# Add watermark
cv2.putText(
    watermarked,
    watermark,
    position,
    font,
    font_scale,
    color,
    thickness,
    cv2.LINE_AA
)

# Display original image
cv2.imshow("Original Image", img)

# Display watermarked image
cv2.imshow("Watermarked Image", watermarked)

# Save watermarked image
cv2.imwrite("./Output/ex26_watermarked.jpg", watermarked)

# Wait for key press
cv2.waitKey(0)
cv2.destroyAllWindows()