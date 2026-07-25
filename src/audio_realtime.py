"""
Real-time microphone speech emotion analysis with CSV logging.

Usage:
    python src\\audio_realtime.py --label patient_a
    python src\\audio_realtime.py --duration 120 --segment-seconds 4 --hop-seconds 1

Output:
    output\\csv\\audio_emotions_<label>_<timestamp>.csv
    output\\reports\\report_audio_emotions_<label>_<timestamp>.html
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from queue import Empty, Queue

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(__file__))
from backend_audio_transformers import DEFAULT_MODEL, EXPRESSIONS, analyze_audio


def parse_args():
    p = argparse.ArgumentParser(description="Real-time microphone emotion analysis")
    p.add_argument("--label", "-l", default=None, help="Session label for output naming")
    p.add_argument("--duration", type=float, default=0, help="Seconds to record; 0 means until Ctrl+C")
    p.add_argument("--segment-seconds", type=float, default=4.0, help="Audio window length per inference")
    p.add_argument("--hop-seconds", type=float, default=1.0, help="Seconds between inferences")
    p.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate")
    p.add_argument("--device", default=None, help="sounddevice input device id/name")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face audio-classification model")
    return p.parse_args()


def make_csv_path(label):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_part = f"_{label}" if label else ""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "csv")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"audio_emotions{label_part}_{ts}.csv")


def print_result(elapsed, result):
    emotions = result["emotions"]
    dominant = result["dominant_emotion"]
    bars = []
    for e in EXPRESSIONS:
        value = emotions.get(e, 0.0)
        if value > 0.5:
            bars.append(f"{e}:{value:4.1f}")
    print(f"{elapsed:7.1f}s  {dominant:9s}  " + "  ".join(bars), flush=True)


def main():
    args = parse_args()
    q = Queue()
    sample_rate = int(args.sample_rate)
    segment_samples = max(1, int(args.segment_seconds * sample_rate))
    hop_samples = max(1, int(args.hop_seconds * sample_rate))
    buffer = np.zeros(0, dtype=np.float32)

    csv_path = make_csv_path(args.label)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["elapsed_s", "wall_time", "frame_idx", "dominant_emotion"]
        + EXPRESSIONS
        + ["n_faces", "confidence", "energy_rms", "model"]
    )
    print(f"Logging to: {csv_path}")
    print("Press Ctrl+C to stop and generate the report.")

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())

    start = time.time()
    segment_idx = 0

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=args.device,
            callback=callback,
        ):
            while True:
                if args.duration and time.time() - start >= args.duration:
                    break

                try:
                    chunk = q.get(timeout=0.2)
                except Empty:
                    continue

                buffer = np.concatenate([buffer, chunk])
                while buffer.size >= segment_samples:
                    window = buffer[:segment_samples]
                    buffer = buffer[hop_samples:]
                    elapsed = time.time() - start
                    result = analyze_audio(window, sample_rate=sample_rate, model_name=args.model)
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
                    csv_file.flush()
                    print_result(elapsed, result)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        csv_file.close()

    print(f"Session saved to: {csv_path}")

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
