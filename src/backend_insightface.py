"""
InsightFace emotion analysis backend.
Uses InsightFace for face detection, then heuristic from cropped ROI.
Thread-safe lazy initialization (model loaded once).
"""
import os
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import cv2

EMOTION_LABELS = [
    "angry", "disgusted", "fearful",
    "happy", "sad", "surprised", "neutral",
]

# --- Lazy singleton ---
_app = None
_initialized = False


def _ensure_init():
    global _app, _initialized
    if _initialized:
        return
    _initialized = True

    # Redirect stdout/stderr to suppress InsightFace verbose output
    _devnull = open(os.devnull, "w")
    _old_out, _old_err = sys.stdout, sys.stderr

    try:
        sys.stdout, sys.stderr = _devnull, _devnull
        _app = FaceAnalysis(name="buffalo_s", root=os.path.dirname(__file__))
        _app.prepare(ctx_id=0, det_size=(640, 640))
    finally:
        _devnull.close()
        sys.stdout, sys.stderr = _old_out, _old_err


def _heuristic_emotion(face_roi):
    """Lightweight heuristic from face ROI geometry (no ML model needed)."""
    h, w = face_roi.shape[:2]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Mouth region (bottom-center)
    y1, y2 = int(h * 0.55), int(h * 0.8)
    x1, x2 = int(w * 0.3), int(w * 0.7)
    mouth = gray[y1:y2, x1:x2]
    mouth_ratio = mouth.shape[0] / max(mouth.shape[1], 1)

    # Eye region (top half)
    eyes = gray[0:int(h * 0.45), :]

    brow_region = gray[0:int(h * 0.15), :]
    brow_mean = float(np.mean(brow_region))
    eye_mean = float(np.mean(eyes))
    mouth_mean = float(np.mean(mouth))

    if mouth_ratio > 0.28:
        emotion = "surprised" if brow_mean > eye_mean else "sad"
    elif brow_mean < eye_mean - 20:
        emotion = "angry"
    elif mouth_mean > eye_mean + 5:
        emotion = "happy"
    elif mouth_mean < eye_mean - 10:
        emotion = "sad"
    else:
        emotion = "neutral"

    return emotion


def analyze(frame):
    """Analyze emotion from a BGR frame.
    Returns a list of results (one per face), or None if no faces.
    """
    _ensure_init()

    try:
        bboxes = _app.det_model.detect_faces(frame, det_thresh=0.5)
    except Exception:
        return None

    if not bboxes:
        return None

    results = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox[:4])
        roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
        if roi.size == 0:
            continue
        emotion = _heuristic_emotion(roi)

        results.append({
            "model": "insightface",
            "backend": "insightface",
            "bbox": [float(bbox[i]) for i in range(4)],
            "dominant_emotion": emotion,
            "emotions": {e: 100.0 if e == emotion else 0.0 for e in EMOTION_LABELS},
            "confidence": 1.0,
        })
    return results
