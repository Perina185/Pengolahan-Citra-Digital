import cv2
import numpy as np

# 1. Ambil gambar dengan noise
# Load gambar asli
image = cv2.imread('wp.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Gambar tidak ditemukan!")
    exit()

# Tambahkan noise ke gambar (Gaussian noise)
noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# Simpan gambar dengan noise untuk melihat perbedaan
cv2.imwrite('noisy_image.jpg', noisy_image)

# 2. Gunakan filter untuk mengurangi noise
# Mean filter
mean_filtered = cv2.blur(noisy_image, (5, 5))

# Median filter
median_filtered = cv2.medianBlur(noisy_image, 5)

# Gaussian filter
gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# 3. Equalisasi histogram untuk meningkatkan kontras
equalized_image = cv2.equalizeHist(gaussian_filtered)

# 4. Transformasi geometris
# Rotasi
(h, w) = equalized_image.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)  # 45 derajat rotasi
rotated_image = cv2.warpAffine(equalized_image, rotation_matrix, (w, h))

# Scaling (resize 1.5x)
scaled_image = cv2.resize(equalized_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# Simpan hasil transformasi
cv2.imwrite('mean_filtered.jpg', mean_filtered)
cv2.imwrite('median_filtered.jpg', median_filtered)
cv2.imwrite('gaussian_filtered.jpg', gaussian_filtered)
cv2.imwrite('equalized_image.jpg', equalized_image)
cv2.imwrite('rotated_image.jpg', rotated_image)
cv2.imwrite('scaled_image.jpg', scaled_image)

# Tampilkan hasil
cv2.imshow("Original Image", image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Mean Filtered", mean_filtered)
cv2.imshow("Median Filtered", median_filtered)
cv2.imshow("Gaussian Filtered", gaussian_filtered)
cv2.imshow("Equalized Image", equalized_image)
cv2.imshow("Rotated Image", rotated_image)
cv2.imshow("Scaled Image", scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
