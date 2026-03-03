#face detection functions
import mediapipe as mp 
import numpy as np
from dataclasses import dataclass

@dataclass
class FaceData:
    bbox: tuple[int, int, int, int]#x,y,w,h
    left_eye: tuple[int, int]#x,y
    right_eye: tuple[int, int]#x,y

class FaceDetector:
    def __init__(self, confidence=0.5, max_faces=10):
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=confidence
        )
        self._max_faces = max_faces
#FaceData for each face in frame
    def detect(self, rgb_frame):
        h,w = rgb_frame.shape[:2]
        results = self._detector.process(rgb_frame)
        faces = []
        if not results.detections:
            return faces
        for detection in results.detections[:self._max_faces]:
            box=detection.location_data.relative_bounding_box
            #to pixel values
            x= max(int(box.xmin*w),0)
            y= max(int(box.ymin*h),0)
            bw= min(int(box.width*w),w - x)
            bh= min(int(box.height*h),h - y)
            kp = detection.location_data.relative_keypoints
            #left eye
            left_eye = (int(kp[0].x*w), int(kp[0].y*h))
            #right eye
            right_eye = (int(kp[1].x*w), int(kp[1].y*h))
            faces.append(FaceData(bbox=(x,y,bw,bh), left_eye=left_eye, right_eye=right_eye))
        return faces
    def close(self):
        self._detector.close()


            
        
    

   
