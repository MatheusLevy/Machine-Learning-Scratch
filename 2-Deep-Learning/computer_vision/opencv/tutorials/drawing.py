import os
import cv2

img = cv2.imread(os.path.join(".", "data", "whiteboard.jpg"))

# Line
cv2.line(img, (30, 30), (120, 140), (0, 255, 0), 3)

# rectangule
cv2.rectangle(img, (200, 350), (450, 500), (0, 0, 255), -1)

# circle
cv2.circle(img, (400, 400), 30, (255, 0, 0), 10)

# text
cv2.putText(img, "Just a Text", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2 ,(255, 255, 0), 3)
cv2.imshow("frame", img)
cv2.waitKey(0)