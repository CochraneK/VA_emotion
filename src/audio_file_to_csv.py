"""
Analyze an audio file and output an emotion CSV/report.

Usage:
    python src\\audio_file_to_csv.py --audio interview.wav --label patient_a
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(__file__))
from backend_audio_transformers import DEFAULT_MODEL, EXPRESSIONS, analyze_audio


def parse_args():
    p = argparse.ArgumentParser(description="Audio file emotion analysis")
    p.add_argument("--audio", required=True, help="Path to wav/flac/ogg audio file")
    p.add_argument("--label", "-l", default=None, help="Session label for output naming")
    p.add_argument("--segment-seconds", type=float, default=4.0, help="Audio window length per inference")
    p.add_argument("--hop-seconds", type=float, default=1.0, help="Seconds between inferences")
    p.add_argument("--sample-rate", type=int, default=16000, help="Analysis sample rate")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face audio-classification model")
    return p.parse_args()


def load_audio(path, target_sr):
    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.5:
        audio = audio / max(peak, 1e-6)
    return audio, sr


def make_csv_path(label):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_part = f"_{label}" if label else ""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "csv")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"audio_emotions{label_part}_{ts}.csv")


def main():
    args = parse_args()
    audio, sr = load_audio(args.audio, args.sample_rate)
    segment_samples = max(1, int(args.segment_seconds * sr))
    hop_samples = max(1, int(args.hop_seconds * sr))

    csv_path = make_csv_path(args.label)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["elapsed_s", "wall_time", "frame_idx", "dominant_emotion"]
            + EXPRESSIONS
            + ["n_faces", "confidence", "energy_rms", "model"]
        )

        segment_idx = 0
        for start in range(0, max(1, audio.size - segment_samples + 1), hop_samples):
            window = audio[start:start + segment_samples]
            if window.size < segment_samples:
                break
            elapsed = start / sr
            result = analyze_audio(window, sample_rate=sr, model_name=args.model)
            segment_idx += 1
            if result is None:
                continue

            emotions = result.get("emotions", {})
            row = [
                round(elapsed, 3),
                datetime.now().isoformat(timespec="milliseconds"),
                segment_idx,
                result.get("dominant_emotion", "?"),
            ]
            row.extend(round(emotions.get(e, 0.0), 1) for e in EXPRESSIONS)
            row.extend([
                1,
                round(float(result.get("confidence", 0.0)), 4),
                round(float(result.get("energy_rms", 0.0)), 6),
                result.get("model", args.model),
            ])
            writer.writerow(row)

            if segment_idx % 10 == 0:
                print(f"  analyzed {segment_idx} windows ({elapsed:.1f}s)")

    print(f"CSV saved to: {csv_path}")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from emotion_report import build_report

    html = build_report(csv_path)
    if html:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "output", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"report_{basename}.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
