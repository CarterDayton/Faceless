import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceDetector


def blur_faces(video_source, scale=0.4, blur_strength=(27, 27)):
    # Load face landmarks
    fl = FaceDetector()

    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        frame_copy = frame.copy()
        height, width, _ = frame.shape

        # 1. Face landmarks detection
        landmarks = fl.get_facial_landmarks(frame)
        if landmarks.size == 0:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        convexhull = cv2.convexHull(landmarks)

        # Face blurring
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, convexhull, 255)

        # Extract the face
        frame_copy = cv2.blur(frame_copy, blur_strength)
        face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

        # Extract background
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Final result
        result = cv2.add(background, face_extracted)

        cv2.imshow("Frame", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def pixelate_faces(video_source, scale=0.4, pixelation_size=(10, 10)):
    # Load face landmarks
    fl = FaceDetector()

    cap = cv2.VideoCapture(video_source)
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        frame_copy = frame.copy()
        height, width, _ = frame.shape

        # 1. Face landmarks detection
        landmarks = fl.get_facial_landmarks(frame)
        if landmarks.size == 0:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        convexhull = cv2.convexHull(landmarks)

        # Face pixelation
        






















    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    blur_faces("stockVid.mp4")
