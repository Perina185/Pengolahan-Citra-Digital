import cv2

# Membaca gambar dalam mode grayscale
image = cv2.imread('hh.jpg', cv2.IMREAD_GRAYSCALE)

# Inisialisasi objek SIFT
sift = cv2.SIFT_create()

# Mendeteksi keypoints dan komputasi deskriptor
keypoints, descriptors = sift.detectAndCompute(image, None)

# Menggambar keypoints di citra
sift_image = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Menampilkan hasil
cv2.imshow('SIFT Features', sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
