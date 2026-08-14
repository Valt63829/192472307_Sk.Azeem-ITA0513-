import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread("railway.jpg")

if image is None:
    print("Error: railway.jpg not found.")
    exit()

original = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# 2. GAUSSIAN SMOOTHING
# ============================================================

blur = cv2.GaussianBlur(gray, (5, 5), 1.5)


# ============================================================
# 3. CANNY EDGE DETECTION
# ============================================================

edges = cv2.Canny(
    blur,
    threshold1=50,
    threshold2=150
)


# ============================================================
# 4. REGION OF INTEREST
# ============================================================

height, width = edges.shape

mask = np.zeros_like(edges)

# Focus mainly on the lower/central railway region
roi_points = np.array([
    [
        (0, height),
        (width, height),
        (int(width * 0.75), int(height * 0.45)),
        (int(width * 0.25), int(height * 0.45))
    ]
], dtype=np.int32)

cv2.fillPoly(mask, roi_points, 255)

roi_edges = cv2.bitwise_and(edges, mask)


# ============================================================
# 5. EXTRACT EDGE POINTS
# ============================================================

ys, xs = np.where(roi_edges > 0)

points = np.column_stack((xs, ys))

print("Total edge points:", len(points))


# ============================================================
# 6. RANSAC ITERATION CALCULATION
# ============================================================

outlier_ratio = 0.40
inlier_probability = 1 - outlier_ratio

confidence = 0.99

# Two points are required to define a line
sample_size = 2

k = math.log(1 - confidence) / \
    math.log(1 - inlier_probability ** sample_size)

iterations = math.ceil(k)

print("\nRANSAC Parameters")
print("-------------------------")
print("Outlier ratio:", outlier_ratio)
print("Inlier probability:", inlier_probability)
print("Confidence:", confidence)
print("Sample size:", sample_size)
print("Calculated iterations:", k)
print("Required iterations:", iterations)


# ============================================================
# 7. LINE DISTANCE FUNCTION
# ============================================================

def point_line_distance(point, line):
    """
    Line equation:
        ax + by + c = 0

    Returns perpendicular distance.
    """

    x, y = point
    a, b, c = line

    distance = abs(a * x + b * y + c)

    denominator = math.sqrt(a * a + b * b)

    if denominator == 0:
        return float("inf")

    return distance / denominator


# ============================================================
# 8. CREATE LINE FROM TWO POINTS
# ============================================================

def create_line(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    # ax + by + c = 0

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    return (a, b, c)


# ============================================================
# 9. RANSAC IMPLEMENTATION
# ============================================================

def ransac_line(points, iterations, threshold):

    if len(points) < 2:
        return None, []

    best_line = None
    best_inliers = []

    for i in range(iterations):

        # Randomly select two points
        sample_indices = random.sample(range(len(points)), 2)

        p1 = points[sample_indices[0]]
        p2 = points[sample_indices[1]]

        # Avoid identical points
        if np.array_equal(p1, p2):
            continue

        # Generate candidate line
        line = create_line(p1, p2)

        # Find inliers
        inliers = []

        for point in points:

            distance = point_line_distance(point, line)

            if distance < threshold:
                inliers.append(point)

        # Keep model with maximum inliers
        if len(inliers) > len(best_inliers):

            best_line = line
            best_inliers = inliers

    return best_line, np.array(best_inliers)


# ============================================================
# 10. RUN RANSAC
# ============================================================

distance_threshold = 3.0

best_line, inliers = ransac_line(
    points,
    iterations,
    distance_threshold
)

print("\nRANSAC Result")
print("-------------------------")

if best_line is None:

    print("No line detected.")

else:

    print("Inliers detected:", len(inliers))
    print("Outliers detected:", len(points) - len(inliers))


# ============================================================
# 11. VISUALIZE RANSAC RESULT
# ============================================================

result = original.copy()

# Draw all edge points as small circles
for point in points:

    x, y = point

    cv2.circle(
        result,
        (int(x), int(y)),
        1,
        (0, 0, 255),
        -1
    )


# ============================================================
# 12. DRAW BEST RANSAC LINE
# ============================================================

if best_line is not None:

    a, b, c = best_line

    # Calculate two points on the line

    x1 = 0
    x2 = width - 1

    if abs(b) > 1e-6:

        y1 = int(-(a * x1 + c) / b)
        y2 = int(-(a * x2 + c) / b)

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Railway Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Smoothed Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(roi_edges, cmap="gray")
plt.title("Canny + ROI")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("RANSAC Rail-Line Detection")
plt.axis("off")

plt.tight_layout()
plt.show()