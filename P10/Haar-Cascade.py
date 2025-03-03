import cv2

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Baca gambar dan ubah menjadi grayscale
image = cv2.imread('ss.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Deteksi wajah
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Gambar kotak di sekitar wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Tampilkan gambar dengan kotak deteksi
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
