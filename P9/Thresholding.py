import cv2
import numpy as np

image = cv2.imread('wp.jpg', cv2.IMREAD_GRAYSCALE)
ret, thresh1 = cv2.threshold(image, 127,255, cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11 , 2)
cv2.imshow('Global Thresholding', thresh1)
cv2.imshow('Adaptive Thresholding', thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()