import cv2
import os

# read image 
image_path = os.path.join(".", "data", "fsociety.png")
img = cv2.imread(image_path)

# Write Image
cv2.imwrite("fsociety2.png", img)

# Visualize Image
cv2.imshow("Fsociety", img)
cv2.waitKey(0)


# Read Video
video_path = os.path.join(".", "data", "toothless_dance.mp4")
video = cv2.VideoCapture(video_path)

# Visualize Video
ret = True
while ret:
    ret, frame = video.read()
    print(ret)
    if ret:
        cv2.imshow("frame", frame)
        cv2.waitKey(40)

video.release()
cv2.destroyAllWindows()

# read webcam
webcam = cv2.VideoCapture(0) # number of the webcam

while True:
    ret, frame = webcam.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(40) & 0xFF == ord("q"):
        break


webcam.release()
cv2.destroyAllWindows()

# Cropping Image
image_path = os.path.join(".", "data", "fsociety.png")
img = cv2.imread(image_path)

cropped_img  = img[70:150, 75:150]

cv2.imshow("crop", cropped_img)
cv2.waitKey(0)
