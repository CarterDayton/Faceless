import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
    
    def get_facial_landmarks(self, image):
        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_image)

        points = []
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                for i in range(0, 468):
                    pt1 = face_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    points.append([x, y])
        return np.array(points, dtype=np.int32)


        
    

   
