import os
import cv2

img_path = os.path.join(".", "data", "land.png")
img = cv2.imread(img_path)
img = cv2.resize(img, (400, 500))

kernel_size = (7, 7)
# simple blur
img_blur =  cv2.blur(img, kernel_size)
img_gaussian_blur = cv2.GaussianBlur(img, kernel_size, 3)
img_median_blur = cv2.medianBlur(img, 7)

cv2.imshow("frame_mediam_blue", img_median_blur)
cv2.imshow("frame_gaussian_blur", img_gaussian_blur)
cv2.imshow("frame_blur", img_blur)
cv2.waitKey(0)