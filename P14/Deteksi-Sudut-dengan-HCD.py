import cv2
import numpy as np

# Baca citra dalam grayscale
image = cv2.imread('wp.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan Harris Corner Detector
gray = np.float32(image)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Ubah citra grayscale menjadi citra berwarna (3 channel)
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Tingkatkan sudut yang terdeteksi dengan memberi warna merah
color_image[corners > 0.01 * corners.max()] = [0, 0, 255]  # Merah (BGR)

# Tampilkan hasil deteksi sudut
cv2.imshow('Harris Corner Detection', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()