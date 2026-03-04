#blur/black bar functions
import cv2
from face_detector import FaceData


def apply_gaussian_blur(frame, faces: list[FaceData], intensity: int) -> None:
    fh, fw = frame.shape[:2]
    k = intensity if intensity % 2 == 1 else intensity + 1
    for face in faces:
        x, y, w, h = face.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)

def apply_black_bar(frame, faces: list[FaceData]) -> None:
    #Draws a black bar over each face bounding box in the frame in-place.
    for face in faces:
        x, y, w, h = face.bbox
        left_eye_x, _ = face.left_eye
        right_eye_x, _ = face.right_eye
        bar_y1 = min(y + h // 4, frame.shape[0])
        bar_y2 = min(y + 2 * h // 4, frame.shape[0])
        bar_x1 = max(min(left_eye_x, right_eye_x) - w // 3, 0)
        bar_x2 = min(max(left_eye_x, right_eye_x) + w // 3, frame.shape[1])
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 0, 0), thickness=-1)
#Pixelates the face regions in the frame
def apply_pixelate(frame, faces: list[FaceData], pixel_size: int) -> None:
    fh, fw = frame.shape[:2]
    for face in faces:
        x, y, w, h = face.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 > x1 and y2 > y1:
            rh, rw = y2 - y1, x2 - x1
            small = cv2.resize(frame[y1:y2, x1:x2], (max(5, rw // pixel_size), max(1, rh // pixel_size)), interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


