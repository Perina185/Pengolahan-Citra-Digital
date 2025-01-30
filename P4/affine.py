import cv2
import numpy as np

image = cv2.imread('wp.jpg')
(h, w) = image.shape[:2]
points1 = np.float32([[50,50],[200,50],[50,200]])
points2 = np.float32([[10,100],[200,50],[100,250]])

M_affine = cv2.getAffineTransform(points1, points2)

affine_transformed_image = cv2.warpAffine(image, M_affine, (w, h))
cv2.imshow('Affine Transformed Image', affine_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()