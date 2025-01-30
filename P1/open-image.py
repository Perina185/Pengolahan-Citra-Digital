import cv2
# Membaca gambar
image = cv2.imread('Tomioka wallpaper.jpeg')
# Menampilkan gambar
cv2.imshow('Display Window', image)
# Menunggu hingga ada input dari keyboard
cv2.waitKey(0)
# Menutup semua jendela
cv2.destroyAllWindows()


import cv2
import numpy as np

def rotate(self, degree):
    height, width, _ = self.image.shape

    rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    cos = np.abs(rotationMatrix[0, 0])
    sin = np.abs(rotationMatrix[0, 1])

    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))

    rotationMatrix[0, 2] += (nW / 2) - width / 2
    rotationMatrix[1, 2] += (nH / 2) - height / 2

    image = cv2.warpAffine(self.image, rotationMatrix, (nW, nH))
    self.image = image
    return self

 