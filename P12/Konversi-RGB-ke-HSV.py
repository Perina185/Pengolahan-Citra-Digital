import cv2

# Membaca citra dalam format RGB
image = cv2.imread('ss.jpg')

# Konversi citra dari RGB ke HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Menampilkan citra hasil konversi
cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
