import cv2

# Membaca gambar
image = cv2.imread('Tomioka wallpaper.jpeg')


# Ukuran baru untuk gambar (misalnya 400x400 piksel)
new_width = 900
new_height = 1600

# Mengubah ukuran gambar
resized_image = cv2.resize(image, (new_width, new_height))

# Menampilkan gambar asli dan gambar yang telah diubah ukurannya
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)

# Menunggu hingga ada input dari keyboard
cv2.waitKey(0)

# Menutup semua jendela
cv2.destroyAllWindows()

# Menyimpan gambar yang diubah ukurannya (jika diperlukan)
cv2.imwrite('resized_image.jpg', resized_image)