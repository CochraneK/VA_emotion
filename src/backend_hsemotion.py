"""
HSEmotion GPU backend — academic FER model.

=== What this does ===
Face detection: OpenCV YuNet (CNN-based, high accuracy, ~5ms CPU)
Emotion model: HSEmotion enet_b0_8_best_afew (Savchenko, CVPR 2022)
    → EfficientNet-B0 backbone trained on AffectNet (400k+ labeled faces)
    → Outputs softmax probabilities across 8 emotion classes
    → Merged to 7 classes: neutral, happy, sad, angry, fearful, disgusted, surprised

=== Model details ===
Paper:  "HSEmotion: High-Speed Emotion Recognition" (Savchenko, 2022)
        https://arxiv.org/abs/2108.01588
Dataset: AffectNet (Mollahosseini et al., 2017)
          → 450,000 face images, 8 expression labels
          → Labels: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
Architecture: EfficientNet-B0 (5.3M params, 224×224 input)
Training:   Cross-entropy on AffectNet, class-balanced sampling
Device:     CUDA (GPU) — ~20ms inference, ~5ms face detection

=== Decision logic ===
- Softmax probabilities → per-class score (0–1)
- "Contempt" merged into "disgusted" (no standalone contempt in 7-class scheme)
- dominant_emotion = argmax over merged scores
- No thresholding / rule-based heuristics — pure model output
- confidence = probability of the dominant class

=== Face detection threshold ===
- YuNet score_threshold = 0.7 (90%+ recall on frontal faces)
- nms_threshold = 0.3
- Real faces typically score >0.85; chairs/walls score <0.3 (reliably filtered)
"""
import os
import numpy as np
import cv2
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer

EXPRESSIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

# HSEmotion 8-class → 7-class (Contempt → Disgusted)
LABEL_MAP = {
    "Anger": "angry", "Contempt": "disgusted", "Disgust": "disgusted",
    "Fear": "fearful", "Happiness": "happy", "Neutral": "neutral",
    "Sadness": "sad", "Surprise": "surprised",
}

_model = None
_detector = None
_device = None

DET_THRESHOLD = 0.65  # YuNet face confidence — typical faces >0.85
NMS_THRESHOLD = 0.3


def _init():
    global _model, _detector, _device
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"HSEmotion device: {_device}")

    # YuNet face detector (OpenCV >= 4.5.4)
    model_path = os.path.join(os.path.dirname(__file__),
                              "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(model_path):
        _download_yunet(model_path)

    if os.path.exists(model_path):
        _detector = cv2.FaceDetectorYN.create(
            model_path, "", (480, 360),
            score_threshold=DET_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            top_k=5000,
        )
        print("Face detector: YuNet (CNN)")
    else:
        _detector = None
        print("Face detector: Haar cascade (fallback)")

    _model = HSEmotionRecognizer(model_name="enet_b0_8_best_afew", device=_device)


def _download_yunet(path):
    import urllib.request
    url = ("https://github.com/opencv/opencv_zoo/raw/main/"
           "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
    try:
        print("Downloading YuNet face detector...")
        urllib.request.urlretrieve(url, path)
        print("Done.")
    except Exception as e:
        print(f"Download failed: {e}")


def analyze(frame):
    try:
        _init()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 1: YuNet face detection (~5ms)
        if _detector is not None:
            _detector.setInputSize((w, h))
            _, faces = _detector.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            # faces: [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
            #         nose_x, nose_y, mouth_right_x, mouth_right_y, mouth_left_x,
            #         mouth_left_y, confidence]
            boxes = [(int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3]))
                     for f in faces]
        else:
            # Fallback: Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            faces_list = cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            if not isinstance(faces_list, np.ndarray) or len(faces_list) == 0:
                return None
            boxes = [(x, y, x + fw, y + fh) for (x, y, fw, fh) in faces_list]

        # Step 2: largest face → crop → emotion (~20ms on GPU)
        largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = largest

        pad = int((y2 - y1) * 0.1)
        fx1, fy1 = max(0, x1 - pad), max(0, y1 - pad)
        fx2, fy2 = min(w, x2 + pad), min(h, y2 + pad)

        face_roi = rgb[fy1:fy2, fx1:fx2]
        if face_roi.size == 0:
            return None

        # predict_emotions(logits=False) → softmax probabilities (sum = 1.0)
        _, scores = _model.predict_emotions(face_roi, logits=False)

        # Merge 8 classes → 7
        merged = {e: 0.0 for e in EXPRESSIONS}
        for idx, hse_label in _model.idx_to_class.items():
            our_label = LABEL_MAP.get(hse_label, "neutral")
            merged[our_label] += float(scores[idx])

        dominant = max(merged, key=merged.get)

        return {
            "model": "hsemotion_enet_b0",
            "backend": "hsemotion",
            "dominant_emotion": dominant,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "emotions": {e: round(merged[e] * 100, 1) for e in EXPRESSIONS},
            "confidence": float(merged[dominant]),
        }
    except Exception as e:
        print(f"Infer error: {e}")
        return None
