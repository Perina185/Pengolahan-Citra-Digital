import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Load gambar dengan objek jelas dan latar belakang berbeda
image = cv2.imread('hh.jpg')
if image is None:
    print("Error: Gambar tidak ditemukan!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Segmentasi berbasis thresholding
_, binary_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 3. Deteksi tepi menggunakan Canny
edges = cv2.Canny(gray, 100, 200)

# 4. Segmentasi berbasis Watershed
# Konversi ke grayscale dan threshold
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological operations untuk noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that the background is 1 instead of 0
markers = markers + 1

# Mark the unknown region with zero
markers[unknown == 255] = 0

# Watershed segmentation
markers = cv2.watershed(image, markers)
watershed_result = image.copy()
watershed_result[markers == -1] = [0, 0, 255]  # Mark boundary with red

# 5. Segmentasi berbasis K-Means Clustering
# Reshape image to 2D array
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define criteria and apply K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3  # Number of clusters
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers to uint8 and labels back to image
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# Save results
cv2.imwrite('binary_thresh.jpg', binary_thresh)
cv2.imwrite('edges.jpg', edges)
cv2.imwrite('watershed_result.jpg', watershed_result)
cv2.imwrite('segmented_image.jpg', segmented_image)

# Display results
titles = ['Original Image', 'Thresholding', 'Canny Edges', 'Watershed', 'K-Means Segmentation']
images = [image, binary_thresh, edges, watershed_result, segmented_image]

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.show()
