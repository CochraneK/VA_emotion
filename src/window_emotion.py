"""Analyze the FasterLivePortrait Render window in real time.

Examples:
    python src/window_emotion.py --layout auto
    python src/window_emotion.py --layout both --label flp_test

Controls: Q/Esc quit, S screenshot, B toggle single/both,
          R return to configured layout, [ and ] adjust split position.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from backend_hsemotion import analyze
from window_capture import DynamicWindowCapture

EMOTIONS = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]
COLORS = {
    "happy": (0, 255, 0), "sad": (255, 100, 100), "angry": (0, 0, 255),
    "surprised": (255, 200, 0), "fearful": (128, 0, 128),
    "disgusted": (0, 128, 0), "neutral": (180, 180, 180),
}
DISPLAY = "Face Emotion - FLP Window"
HEADER = ["elapsed_s", "wall_time", "frame_idx", "dominant_emotion", *EMOTIONS, "n_faces"]


def args_parser():
    p = argparse.ArgumentParser(description="Capture and analyze the FLP Render window")
    p.add_argument("--window-title", default="Render")
    p.add_argument("--layout", choices=["auto", "single", "both"], default="auto")
    p.add_argument("--split-ratio", type=float, default=.5)
    p.add_argument("--label", default="flp")
    p.add_argument("--skip", type=int, default=3)
    p.add_argument("--analysis-width", type=int, default=720)
    p.add_argument("--auto-both-ratio", type=float, default=2.35)
    a = p.parse_args()
    a.skip = max(1, a.skip)
    a.analysis_width = max(240, a.analysis_width)
    a.split_ratio = float(np.clip(a.split_ratio, .2, .8))
    return a


def normalize(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and raw:
        return raw[0] if isinstance(raw[0], dict) else None
    return None


def infer(frame):
    try:
        return normalize(analyze(frame))
    except Exception as exc:
        print(f"Infer error: {exc}")
        return None


def resize_max(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def split_both(frame, ratio):
    x = int(np.clip(round(frame.shape[1] * ratio), 1, frame.shape[1] - 1))
    return frame[:, :x].copy(), frame[:, x:].copy()


def emotion_vector(result):
    if not result:
        return None
    scores = result.get("emotions", {})
    return np.array([float(scores.get(e, 0)) for e in EMOTIONS], dtype=float)


def similarity(a, b):
    va, vb = emotion_vector(a), emotion_vector(b)
    if va is None or vb is None:
        return None
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 1e-9 else None


def draw_face(frame, result, title):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1, cv2.LINE_AA)
    if not result:
        cv2.putText(out, "No face", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, .55, (150, 150, 150), 1)
        return out
    emo = result.get("dominant_emotion", "?")
    color = COLORS.get(emo, (255, 255, 255))
    box = result.get("bbox")
    if box and len(box) >= 4:
        x1, y1, x2, y2 = map(lambda v: int(round(float(v))), box[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{emo} {float(result.get('confidence', 0)):.0%}",
                    (max(0, x1), max(48, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 2)
    return out


def draw_chart(height, width, result, title):
    c = np.full((max(170, height), max(200, width), 3), 25, np.uint8)
    cv2.putText(c, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .47, (210, 210, 210), 1)
    scores = result.get("emotions", {}) if result else {}
    top, gap = 34, 3
    bh = max(10, (c.shape[0] - 58 - gap * 6) // 7)
    left, right = 68, c.shape[1] - 42
    for i, emo in enumerate(EMOTIONS):
        y = top + i * (bh + gap)
        val = float(np.clip(scores.get(emo, 0), 0, 100))
        color = COLORS[emo]
        cv2.putText(c, emo[:5], (4, y + bh - 2), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1)
        cv2.rectangle(c, (left, y), (right, y + bh), (48, 48, 48), -1)
        fill = int((right - left) * val / 100)
        cv2.rectangle(c, (left, y), (left + fill, y + bh), color, -1)
        cv2.putText(c, f"{val:.0f}%", (right + 3, y + bh - 2), cv2.FONT_HERSHEY_SIMPLEX, .29, color, 1)
    return c


def panel(frame, result, title):
    face = draw_face(frame, result, title)
    chart = draw_chart(max(180, int(frame.shape[0] * .42)), frame.shape[1], result, f"{title} emotions")
    return np.vstack([face, chart])


def log_row(writer, elapsed, idx, result):
    if not result:
        return False
    scores = result.get("emotions", {})
    writer.writerow([round(elapsed, 3), datetime.now().isoformat(timespec="milliseconds"), idx,
                     result.get("dominant_emotion", "?"),
                     *[round(float(scores.get(e, 0)), 1) for e in EMOTIONS], 1])
    return True


def make_logger(csv_dir, label, role, stamp):
    path = csv_dir / f"emotions_{label}_{role}_{stamp}.csv"
    f = path.open("w", newline="", encoding="utf-8")
    w = csv.writer(f); w.writerow(HEADER)
    print(f"Logging {role}: {path}")
    return path, f, w


def create_reports(paths):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from emotion_report import build_report
        reports = ROOT / "output" / "reports"; reports.mkdir(parents=True, exist_ok=True)
        for path in paths.values():
            html = build_report(str(path))
            if html:
                (reports / f"report_{path.stem}.html").write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"Report generation skipped: {exc}")
    if "driving" in paths and "render" in paths:
        try:
            subprocess.run([sys.executable, str(ROOT / "tools" / "emotion_sync.py"),
                            str(paths["driving"]), str(paths["render"])], cwd=ROOT, check=False)
        except Exception as exc:
            print(f"Sync report skipped: {exc}")


def main():
    a = args_parser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = ROOT / "output" / "csv"; csv_dir.mkdir(parents=True, exist_ok=True)
    source = DynamicWindowCapture(a.window_title)
    cv2.namedWindow(DISPLAY, cv2.WINDOW_NORMAL); cv2.resizeWindow(DISPLAY, 1100, 650)
    files, handles, writers, rows = {}, {}, {}, {}
    last = {"window": None, "driving": None, "render": None}
    override = None; idx = 0; start = time.time()
    print("Controls: Q quit | S screenshot | B toggle layout | R reset | [ ] split")
    try:
        while True:
            ok, captured, changed = source.read()
            if not ok:
                wait = np.full((180, 720, 3), 24, np.uint8)
                cv2.putText(wait, f"Waiting for window: {a.window_title}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (220, 220, 220), 1)
                cv2.putText(wait, "Start FLP and keep Render visible. Q quits.", (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (160, 160, 160), 1)
                cv2.imshow(DISPLAY, wait)
                if cv2.waitKey(50) & 0xFF in (ord('q'), 27): break
                continue
            idx += 1
            ratio = captured.shape[1] / max(1, captured.shape[0])
            layout = override or (a.layout if a.layout != "auto" else ("both" if ratio >= a.auto_both_ratio else "single"))
            frame = resize_max(captured, a.analysis_width * (2 if layout == "both" else 1))
            roles = ["driving", "render"] if layout == "both" else ["window"]
            parts = split_both(frame, a.split_ratio) if layout == "both" else [frame]
            if (idx - 1) % a.skip == 0:
                for role, part in zip(roles, parts):
                    last[role] = infer(part)
                    if role not in writers:
                        path, f, w = make_logger(csv_dir, a.label, role, stamp)
                        files[role], handles[role], writers[role], rows[role] = path, f, w, 0
                    if log_row(writers[role], time.time() - start, idx, last[role]):
                        handles[role].flush(); rows[role] += 1
            if layout == "both":
                left, right = panel(parts[0], last["driving"], "Driving / Real"), panel(parts[1], last["render"], "Render / Avatar")
                display = np.hstack([left, right])
                footer = np.full((34, display.shape[1], 3), 18, np.uint8)
                sim = similarity(last["driving"], last["render"])
                match = bool(last["driving"] and last["render"] and last["driving"].get("dominant_emotion") == last["render"].get("dominant_emotion"))
                text = f"similarity: {sim:.1%} | dominant match: {'yes' if match else 'no'} | split {a.split_ratio:.2f}" if sim is not None else f"both | waiting for two faces | split {a.split_ratio:.2f}"
                cv2.putText(footer, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 220, 220), 1)
                display = np.vstack([display, footer])
            else:
                display = panel(parts[0], last["window"], "Captured window")
            cv2.imshow(DISPLAY, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            if key == ord('s'):
                cv2.imwrite(str(csv_dir / f"window_screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"), display)
            elif key == ord('b'): override = "single" if layout == "both" else "both"
            elif key == ord('r'): override = None
            elif key == ord('['): a.split_ratio = float(np.clip(a.split_ratio - .01, .2, .8))
            elif key == ord(']'): a.split_ratio = float(np.clip(a.split_ratio + .01, .2, .8))
    finally:
        source.release()
        for f in handles.values(): f.close()
        cv2.destroyAllWindows()
    for role, path in files.items(): print(f"Saved {role} ({rows[role]} rows): {path}")
    create_reports(files)


if __name__ == "__main__":
    main()
