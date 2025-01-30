import cv2

# Membaca citra dalam format RGB
image = cv2.imread('ss.jpg')

# Konversi citra dari RGB ke YCbCr
ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Ekstrak channel Y (luminance)
Y_channel = ycbcr_image[:, :, 0]

# Menampilkan channel Y
cv2.imshow('Y Channel', Y_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()
