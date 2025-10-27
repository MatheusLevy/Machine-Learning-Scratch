import os
import cv2

img_path = os.path.join(".", "data", "land.png")
img = cv2.imread(img_path)
img = cv2.resize(img, (400, 500))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("frame_bgr", img)
cv2.imshow("frame_rgb", img_rgb)
cv2.imshow("frame_gray", img_gray)
cv2.imshow("frame_hsv", img_hsv)
cv2.waitKey(0)
