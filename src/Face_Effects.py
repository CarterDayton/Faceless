import cv2
import numpy as np

# MediaPipe FaceMesh indices for eye corners and top/bottom of each eye.
EYE_INDICES = [33, 133, 362, 263, 159, 145, 386, 374]

_prev_landmarks = None


def _composite(frame, modified, mask):
    face = cv2.bitwise_and(modified, modified, mask=mask)
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    return cv2.add(background, face)


def gaussian_blur(frame, landmarks, mask, cfg):
    _, _, w, _ = cv2.boundingRect(landmarks)
    base = getattr(cfg, "BLUR_INTENSITY", 27)
    k = max(3, int(base * (w / 200.0)))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return _composite(frame, blurred, mask)


def pixelate(frame, landmarks, mask, cfg):
    x, y, w, h = cv2.boundingRect(landmarks)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return frame
    block = max(1, getattr(cfg, "PIXEL_SIZE", 15))
    sw = max(1, w // block)
    sh = max(1, h // block)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    out[y:y + h, x:x + w] = pix
    return _composite(frame, out, mask)


def eye_bar(frame, landmarks, _mask, cfg):
    eye_pts = landmarks[EYE_INDICES]
    ex, ey, ew, eh = cv2.boundingRect(eye_pts)
    _, _, _, face_h = cv2.boundingRect(landmarks)

    bar_h = max(eh, int(face_h * getattr(cfg, "EYE_BAR_HEIGHT_RATIO", 0.25)))
    cy = ey + eh // 2
    pad_x = int(ew * 0.1)

    y1 = max(0, cy - bar_h // 2)
    y2 = min(frame.shape[0], cy + bar_h // 2)
    x1 = max(0, ex - pad_x)
    x2 = min(frame.shape[1], ex + ew + pad_x)

    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2),
                  getattr(cfg, "EYE_BAR_COLOR", (0, 0, 0)), -1)
    return out


# Add a new filter by writing a `(frame, landmarks, mask, cfg) -> frame`
# function and registering it here.
FILTERS = {
    "blur": gaussian_blur,
    "pixelate": pixelate,
    "eye_bar": eye_bar,
}
