import cv2

# Membuat detektor ORB
orb = cv2.ORB_create()

# Load gambar
image = cv2.imread('hh.jpg')

# Deteksi keypoints dan deskriptor
keypoints, descriptors = orb.detectAndCompute(image, None)

# Gambarkan keypoints
img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Tampilkan hasilnya
cv2.imshow("Keypoints ORB", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
