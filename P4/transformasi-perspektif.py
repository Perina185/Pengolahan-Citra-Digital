import cv2
import numpy as np

image = cv2.imread('wp.jpg')

points1 = np.float32([[56, 65], [386,52], [20,387], [389,390]])
points2 = np.float32([[0,0], [300,0], [0,300], [300,300]])
M_perspactive = cv2.getPerspectiveTransform(points1, points2)
perspective_transformed_image = cv2.warpPerspective(image, M_perspactive, (300,300))

cv2.imshow('perspective Transformed image', perspective_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()