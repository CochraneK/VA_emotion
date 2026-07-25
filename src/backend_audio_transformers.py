"""
Speech emotion backend based on Hugging Face Transformers.

Default model:
    superb/wav2vec2-base-superb-er

The default model predicts four IEMOCAP-style categories
angry / happy / sad / neutral. The CSV schema keeps the same seven emotion
columns as the face pipeline, so missing categories are filled with 0.
"""
import os

import numpy as np

EXPRESSIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
DEFAULT_MODEL = os.environ.get("AUDIO_EMOTION_MODEL", "superb/wav2vec2-base-superb-er")

LABEL_MAP = {
    "neutral": "neutral",
    "neu": "neutral",
    "happy": "happy",
    "hap": "happy",
    "excited": "happy",
    "exc": "happy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
    "ang": "angry",
    "frustrated": "angry",
    "fru": "angry",
    "fear": "fearful",
    "fearful": "fearful",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "surprise": "surprised",
    "surprised": "surprised",
}

_pipe = None
_model_name = None
_device_label = None


def _init(model_name=DEFAULT_MODEL):
    global _pipe, _model_name, _device_label
    if _pipe is not None and _model_name == model_name:
        return

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    _device_label = "cuda" if device == 0 else "cpu"
    print(f"Audio emotion model: {model_name} ({_device_label})")
    _pipe = pipeline("audio-classification", model=model_name, device=device)
    _model_name = model_name


def _normalize_label(label):
    key = str(label).lower().strip()
    if "_" in key:
        key = key.split("_")[-1]
    return LABEL_MAP.get(key, LABEL_MAP.get(key.replace(" ", ""), key))


def analyze_audio(waveform, sample_rate=16000, model_name=DEFAULT_MODEL):
    """
    Return a face-pipeline-compatible emotion dict for a mono audio segment.

    waveform: 1-D numpy array, preferably float32 in [-1, 1].
    sample_rate: sampling rate of waveform.
    """
    _init(model_name)

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return None

    # Avoid clipping surprises from int-like input.
    peak = float(np.max(np.abs(audio)))
    if peak > 1.5:
        audio = audio / max(peak, 1e-6)

    inputs = {"array": audio, "sampling_rate": int(sample_rate)}
    try:
        raw = _pipe(inputs, top_k=None)
    except TypeError:
        raw = _pipe(inputs)

    if raw and isinstance(raw[0], list):
        raw = raw[0]

    scores = {e: 0.0 for e in EXPRESSIONS}
    for item in raw or []:
        label = _normalize_label(item.get("label", "neutral"))
        if label in scores:
            scores[label] += float(item.get("score", 0.0)) * 100.0

    if not any(scores.values()):
        return None

    dominant = max(scores, key=scores.get)
    confidence = scores[dominant] / 100.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

    return {
        "model": model_name,
        "backend": "transformers_audio",
        "dominant_emotion": dominant,
        "emotions": {e: round(scores[e], 1) for e in EXPRESSIONS},
        "confidence": confidence,
        "energy_rms": rms,
    }
