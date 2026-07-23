"""
MediaPipe emotion analysis backend.
Uses solutions API (face_detection + face_mesh).
Returns continuous emotion intensity scores (0-100) for real-time bar display.
Heuristic-based: strong signals needed to override neutral.
"""
import numpy as np
import cv2
from mediapipe.python.solutions import face_detection as mp_fd
from mediapipe.python.solutions import face_mesh as mp_fm

EXPRESSIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


def analyze(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        with mp_fd.FaceDetection(min_detection_confidence=0.5) as detector:
            det_results = detector.process(rgb)
        if not det_results.detections:
            return None

        detection = det_results.detections[0]
        bbox_rel = detection.location_data.relative_bounding_box
        x1 = max(0, int(bbox_rel.xmin * w))
        y1 = max(0, int(bbox_rel.ymin * h))
        x2 = min(w, int((bbox_rel.xmin + bbox_rel.width) * w))
        y2 = min(h, int((bbox_rel.ymin + bbox_rel.height) * h))

        with mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        ) as mesh:
            mesh_results = mesh.process(rgb)

        if not mesh_results.multi_face_landmarks:
            return None

        landmarks = mesh_results.multi_face_landmarks[0]
        coords = np.array([(lm.x, lm.y) for lm in landmarks.landmark])

        # --- Key landmarks ---
        # Mouth
        mouth_top = coords[13]       # upper lip
        mouth_bottom = coords[14]    # lower lip
        mouth_left = coords[61]      # left corner
        mouth_right = coords[291]    # right corner

        # Eyes: upper/lower eyelid pairs
        eye_l_top = coords[159]
        eye_l_bot = coords[145]
        eye_r_top = coords[386]
        eye_r_bot = coords[374]

        # Eyebrows
        brow_l = coords[105]
        brow_r = coords[334]

        # --- Continuous metrics ---
        mouth_h = float(abs(mouth_top[1] - mouth_bottom[1]))
        mouth_w = float(abs(mouth_right[0] - mouth_left[0]))
        mouth_openness = mouth_h / max(mouth_w, 0.001)

        # Smile curve: positive = corners above center (smile), negative = frown
        mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
        corner_avg_y = (mouth_left[1] + mouth_right[1]) / 2
        mouth_curve = float(mouth_center_y - corner_avg_y)
        curve_norm = mouth_curve / max(mouth_w, 0.001)

        # Eye openness
        eye_h_l = float(abs(eye_l_top[1] - eye_l_bot[1]))
        eye_h_r = float(abs(eye_r_top[1] - eye_r_bot[1]))
        eye_openness = (eye_h_l + eye_h_r) / 2

        # Brow position: positive = brows lower than eyes (angry)
        brow_avg_y = (brow_l[1] + brow_r[1]) / 2
        eye_avg_y = (eye_l_top[1] + eye_l_bot[1] + eye_r_top[1] + eye_r_bot[1]) / 4
        brow_norm = float(brow_avg_y - eye_avg_y) / max(mouth_w, 0.001)

        # --- Map metrics to emotion scores (0-100) ---
        scores = {e: 0.0 for e in EXPRESSIONS}

        # Surprised: wide-open mouth + wide eyes
        s_mouth = min(1.0, max(0.0, (mouth_openness - 0.10) / 0.25))
        s_eye = min(1.0, max(0.0, (eye_openness - 0.015) / 0.025))
        scores["surprised"] = round(min(100, (s_mouth * 0.7 + s_eye * 0.3) * 100), 1)

        # Happy: mouth corners pulled up (smile)
        scores["happy"] = round(min(100, max(0.0, (curve_norm - 0.02) / 0.15) * 100), 1)

        # Sad: mouth corners pulled down (frown)
        scores["sad"] = round(min(100, max(0.0, (-curve_norm - 0.02) / 0.15) * 100), 1)

        # Angry: brows pulled down toward eyes
        scores["angry"] = round(min(100, max(0.0, (brow_norm - 0.01) / 0.08) * 100), 1)

        # Fearful: moderate mouth open + wide eyes
        f_mouth = min(1.0, max(0.0, (mouth_openness - 0.05) / 0.2))
        f_eye = min(1.0, max(0.0, (eye_openness - 0.012) / 0.02))
        scores["fearful"] = round(min(100, (f_mouth * 0.5 + f_eye * 0.5) * 80), 1)

        # Disgusted: hard to detect from landmarks alone
        scores["disgusted"] = 0.0

        # Neutral: inverse of strongest emotion — dominates when nothing signals
        max_other = max(scores["happy"], scores["sad"], scores["angry"],
                        scores["surprised"], scores["fearful"], scores["disgusted"])
        scores["neutral"] = round(max(0.0, 100 - max_other), 1)

        # Re-pick dominant after neutral calculation
        dominant = max(scores, key=scores.get)

        # Force neutral when all signals are weak
        if max_other < 15:
            scores["neutral"] = round(100 - max_other, 1)
            dominant = "neutral"

        return {
            "model": "mediapipe",
            "backend": "mediapipe",
            "dominant_emotion": dominant,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "emotions": scores,
            "confidence": float(scores[dominant]) / 100.0,
            "face_landmarks": len(coords),
        }
    except Exception:
        return None
