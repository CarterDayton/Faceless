import cv2
from face_detector import FaceDetector
from filters import apply_gaussian_blur, apply_black_bar, apply_pixelate
from config import CAMERA_INDEX, MAX_FACES, DETECTION_CONFIDENCE, BLUR_INTENSITY, PIXEL_SIZE

# Initialize face detector and webcam
detector = FaceDetector(
    confidence=DETECTION_CONFIDENCE,
    max_faces=MAX_FACES
)
cap = cv2.VideoCapture(CAMERA_INDEX)
blur_enabled = False
black_bar_enabled = False
pixelate_enabled = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect(rgb)

    if blur_enabled and faces:
        apply_gaussian_blur(frame, faces, BLUR_INTENSITY)
    if black_bar_enabled and faces:
        apply_black_bar(frame, faces)
    if pixelate_enabled and faces:
        apply_pixelate(frame, faces, PIXEL_SIZE)

    # Show the live webcam feed
    cv2.imshow("Webcam", frame)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        blur_enabled = not blur_enabled
    elif key == ord("b"):
        black_bar_enabled = not black_bar_enabled
    elif key == ord("p"):
        pixelate_enabled = not pixelate_enabled
    

cap.release()
cv2.destroyAllWindows()
