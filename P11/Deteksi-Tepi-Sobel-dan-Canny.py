import cv2
import matplotlib.pyplot as plt

# Fungsi untuk membaca citra
def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Fungsi untuk deteksi tepi menggunakan Sobel dan Canny
def detect_edges(image):
    # Deteksi tepi Sobel
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y
    sobel_combined = cv2.magnitude(sobelx, sobely)
    
    # Deteksi tepi Canny
    canny_edges = cv2.Canny(image, 100, 200)
    
    return sobel_combined, canny_edges

image_path1 = 'dk.jpg'
image_path2 = 'dh.jpg'

# Baca citra dan terapkan deteksi tepi
image1 = read_image(image_path1)
image2 = read_image(image_path2)

sobel1, canny1 = detect_edges(image1)
sobel2, canny2 = detect_edges(image2)

# Tampilkan hasil untuk citra pertama
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1), plt.imshow(sobel1, cmap='gray'), plt.title('Sobel - Image 1')
plt.subplot(2, 2, 2), plt.imshow(canny1, cmap='gray'), plt.title('Canny - Image 1')
plt.subplot(2, 2, 3), plt.imshow(sobel2, cmap='gray'), plt.title('Sobel - Image 2')
plt.subplot(2, 2, 4), plt.imshow(canny2, cmap='gray'), plt.title('Canny - Image 2')
plt.tight_layout()
plt.show()
