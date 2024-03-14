import os
import cv2

img = cv2.imread(os.path.join(".", "data", "dog.jpg"))
img = cv2.resize(img, (600, 450))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY)

tresh = cv2.blur(thresh, (10, 10))

ret, thresh = cv2.threshold(thresh, 70, 255, cv2.THRESH_BINARY)
#cv2.imshow("frame", img_gray)
#cv2.imshow("thresh", thresh)
#cv2.waitKey(0)

# Adaptative Threshold

thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)
cv2.imshow("frame", thresh)
cv2.waitKey(0)