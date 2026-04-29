import cv2
import numpy as np

import config as cfg
from Face_Effects import FILTERS
from facial_landmarks import FaceDetector

SCALE = 1.0
KEYMAP = {
    ord('1'): "blur",
    ord('2'): "pixelate",
    ord('3'): "eye_bar",
}


def main():
    fl = FaceDetector()
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    effect = "blur"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
        height, width, _ = frame.shape

        landmarks = fl.get_facial_landmarks(frame)
        if landmarks.size > 0:
            hull = cv2.convexHull(landmarks)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            frame = FILTERS[effect](frame, landmarks, mask, cfg)

        cv2.imshow("Faceless", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in KEYMAP:
            effect = KEYMAP[key]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
