import os
import cv2
import mediapipe as mp
import argparse

def process_img(img, face_detection):
    out = face_detection.process(img)
    H, W, _ = img.shape
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w =  int(w * W)
            h = int (h * H)

            #blur faces
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (50, 50))
    return img

# argparse
args = argparse.ArgumentParser()
args.add_argument("--mode", default= "video")
args.add_argument("--filePath", default=r".\data\toothless_dance.mp4")
args = args.parse_args()

#detect faces
mp_face_detector = mp.solutions.face_detection

with mp_face_detector.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)
        # save image
        cv2.imwrite("annonimized.png", img)

    elif args.mode in ["video"]:
    
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter("video_anonimazed.mp4",
                                       cv2.VideoWriter_fourcc(*"MP4V"),
                                       25,
                                       (frame.shape[1], frame.shape[0]))
        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()
        cap.release()
        output_video.release()

    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow("frame", frame)
            cv2.waitKey(25)
            ret, frame = cap.read()
            if cv2.waitKey(40) & 0xFF == ord("q"):
                break

        cap.release()