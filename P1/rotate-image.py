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

# Import the Python Image Processing Library
from PIL import Image

# Giving the original image directory
Original_Image = Image.open("Tomioka wallpaper.jpeg")

# Rotate image by 180 degrees
rotated_image1 = Original_Image.rotate(180)

# Alternative syntax to rotate the image by 90 degrees
rotated_image2 = Original_Image.transpose(Image.ROTATE_90)

# Rotate image by 60 degrees
rotated_image3 = Original_Image.rotate(60)

# Show the rotated images
rotated_image1.show()
rotated_image2.show()
rotated_image3.show()