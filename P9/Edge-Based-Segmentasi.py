import cv2 
Image = cv2.imread('wp.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(Image, 100, 200)
cv2.imshow('Edges Detected', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()