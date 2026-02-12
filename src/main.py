import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# Open the default webcam (0). Change to 1 if you want another camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the live webcam feed
    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
