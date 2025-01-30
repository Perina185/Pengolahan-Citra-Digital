import cv2

# Baca citra dalam mode grayscale
image = cv2.imread('ss.jpg', cv2.IMREAD_GRAYSCALE)

# Deteksi tepi menggunakan metode Canny
edges = cv2.Canny(image, 100, 200)

# Tampilkan hasil deteksi tepi
cv2.imshow('Canny Edge Detection', edges)

# Tunggu input tombol dan tutup jendela
cv2.waitKey(0)
cv2.destroyAllWindows()