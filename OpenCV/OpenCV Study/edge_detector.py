import os
import cv2
import numpy as np

img = cv2.imread(os.path.join(".", "data", "dog.jpg"))
img = cv2.resize(img, (600, 450))

img_edge = cv2.Canny(img, 100, 200)

img_edge_d = cv2.dilate(img_edge, np.ones((2,2), dtype=np.int8))

img_edge_e = cv2.erode(img_edge_d, np.ones((2,2), dtype=np.int8))

cv2.imshow("edges_canny", img_edge)
cv2.imshow("dilated", img_edge_d)
cv2.imshow("erode", img_edge_e)
cv2.waitKey(0)