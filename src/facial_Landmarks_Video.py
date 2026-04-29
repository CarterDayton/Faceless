import cv2
import mediapipe as mp
import numpy as np

#Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def facial_landmarks(image):
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #face landmarks
    result = face_mesh.process(rgb_image)

    points = []
    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = face.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                points.append([x, y])
                cv2.circle(image, (x, y), 1, (100, 100, 0), -1)

    return np.array(points, dtype=np.int32)

if __name__ == "__main__":
    cap = cv2.VideoCapture("stockVid.mp4")

    while True:
        #image == frame
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.resize(image, None, fx=0.4, fy=0.4)

        facial_landmarks(image)

        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
