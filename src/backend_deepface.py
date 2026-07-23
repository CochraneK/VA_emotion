"""
DeepFace emotion analysis backend.
Optional dependency - requires tensorflow.
Falls back gracefully if tensorflow is not installed.
Supports: VGG-Face, Facenet, OpenFace, ArcFace, SFace, Dlib.
"""
import sys

try:
    from deepface import DeepFace
    from deepface.commons.logger import Logger

    DEEPFACE_AVAILABLE = True
    _log = Logger()
    _log.disable()

    MODEL_MAP = {
        "VGG-Face": "VGG-Face",
        "Facenet": "Facenet",
        "OpenFace": "OpenFace",
        "Facenet512": "Facenet512",
        "ArcFace": "ArcFace",
        "SFace": "SFace",
        "Dlib": "Dlib",
    }

    def analyze(frame, model_name="VGG-Face"):
        try:
            result = DeepFace.analyze(
                img=frame,
                actions=["emotion"],
                model_name=MODEL_MAP.get(model_name, "VGG-Face"),
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emotions = result.get("emotion", {})
            if emotions:
                dominant = max(emotions, key=emotions.get)
                return {
                    "model": model_name,
                    "backend": "deepface",
                    "dominant_emotion": dominant,
                    "emotions": dict(emotions),
                    "confidence": float(emotions[dominant]) / 100.0,
                }
        except Exception:
            pass
        return None

except ImportError:
    DEEPFACE_AVAILABLE = False

    def analyze(frame, model_name="VGG-Face"):
        return None
