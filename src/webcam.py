"""
Real-time webcam face emotion detection with live emotion bars and CSV logging.
GPU-accelerated (CUDA) face detection + HSEmotion model.

Usage:
    python webcam.py                      # hsemotion (GPU)
    python webcam.py --label person_a     # tag the recording session

Output: output/emotions_<label>_<timestamp>.csv
"""
import sys
import os
import time
import csv
from collections import deque
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from backend_hsemotion import analyze  # GPU backend

# ---- Config ------------------------------------------------------ #
SKIP = 3          # analyze every N frames (model at 20ms on GPU)
CAM_W = 480       # capture width (keeps GPU load low)
# ------------------------------------------------------------------ #

EMOTION_COLORS = {
    "happy":      (0, 255, 0),
    "sad":        (255, 100, 100),
    "angry":      (0, 0, 255),
    "surprised":  (255, 200, 0),
    "fearful":    (128, 0, 128),
    "disgusted":  (0, 128, 0),
    "neutral":    (180, 180, 180),
}
EMOTION_ORDER = list(EMOTION_COLORS.keys())

# --- parse args ---
session_label = None
for a in sys.argv[1:]:
    if a in ("--label", "-l"):
        session_label = ""
    elif session_label == "":
        session_label = a

# --- CSV ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
label_part = f"_{session_label}" if session_label else ""
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
csv_dir = os.path.join(output_dir, "csv")
os.makedirs(csv_dir, exist_ok=True)

# ensure Plotly is available offline
plotly_js = os.path.join(output_dir, "plotly-2.32.0.min.js")
if not os.path.exists(plotly_js):
    print("Downloading Plotly for offline use...")
    import urllib.request
    d = urllib.request.urlopen(
        "https://registry.npmmirror.com/plotly.js/2.32.0/files/dist/plotly.min.js",
        timeout=30).read()
    with open(plotly_js, "wb") as f:
        f.write(d)
    print(f"Plotly downloaded ({len(d)} bytes)")

csv_path = os.path.join(csv_dir, f"emotions{label_part}_{ts}.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["elapsed_s", "wall_time", "frame_idx", "dominant_emotion"] + EMOTION_ORDER + ["n_faces"])
print(f"Logging to: {csv_path}")

# --- webcam ---
def open_camera():
    # DShow is the only reliable backend on this system; MSMF always fails with -1072875772
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAM_W * 3 / 4))
        ok = False
        for i in range(120):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                ok = True
                return cap, frame
        if not ok:
            print(f"  Camera {idx} warmup failed (120 frames).")
        cap.release()
        time.sleep(0.3)
    return None, None

CAP, first_frame = open_camera()
if CAP is None:
    csv_file.close()
    sys.exit("Error: cannot open webcam")

CHART_W = max(320, min(480, int(first_frame.shape[1] * 0.7)))
emotion_history = deque(maxlen=20)

frame_idx = 0
last_result = None
last_logged = None
fps_text = 0.0
fps_counter = 0
fps_last = time.time()
start_time = time.time()
infer_time = 0.0
infer_count = 0
bad_read_count = 0  # consecutive failed reads — kill camera after 20 failures
MAX_IDLE = 30.0     # seconds without a successful inference before exit
idle_start = 0.0

print("Controls: 'q'=quit | 's'=screenshot")

# Initialize GUI window early so the event loop is ready before heavy GPU work
cv2.namedWindow("Face Emotion", cv2.WINDOW_NORMAL)
# Small delay to let the window manager create the window
time.sleep(0.3)

while True:
    ret, frame = CAP.read()
    if not ret or frame is None or frame.size == 0:
        bad_read_count += 1
        if bad_read_count > 20:
            print(f"Camera lost after {frame_idx} frames. Shutting down.")
            break
        continue
    bad_read_count = 0

    frame_idx += 1
    if frame_idx % 30 == 1:
        print(f"  frame {frame_idx}...", flush=True)

    if not idle_start:
        idle_start = time.time()  # start idle timer on first successful read

    h, w = frame.shape[:2]
    elapsed = time.time() - start_time

    # --- analyze ---
    if frame_idx % SKIP == 1:
        t0 = time.time()
        try:
            raw = analyze(frame)
        except Exception as e:
            print(f"Infer error: {e}")
            raw = None
        idle_start = time.time()
        dt = time.time() - t0
        infer_time += dt
        infer_count += 1

        if raw is None:
            emotion_history.append(None)
        elif isinstance(raw, list):
            last_result = raw
            emotion_history.append(raw[0] if raw else None)
        else:
            last_result = [raw]
            emotion_history.append(raw)

        # CSV
        entry = emotion_history[-1]
        if entry is not None and entry is not last_logged:
            emotions = entry.get("emotions", {})
            row = [round(elapsed, 3), datetime.now().isoformat(timespec="milliseconds"),
                   frame_idx, entry.get("dominant_emotion", "?")]
            for e in EMOTION_ORDER:
                row.append(round(emotions.get(e, 0), 1))
            row.append(len(last_result) if isinstance(last_result, list) else 1)
            csv_writer.writerow(row)
            csv_file.flush()
            last_logged = entry

    # --- face overlay ---
    display = frame.copy()
    if last_result:
        items = last_result if isinstance(last_result, list) else [last_result]
        for r in items:
            emotion = r.get("dominant_emotion", "?")
            conf = r.get("confidence", 0)
            label = f"{emotion} ({conf:.1%})"
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            if "bbox" in r:
                bx = [int(r["bbox"][i]) for i in range(4)]
                cv2.rectangle(display, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                cv2.putText(display, label, (bx[0], bx[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if session_label:
        cv2.putText(display, f"[{session_label}]", (w - 130, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # --- FPS ---
    fps_counter += 1
    now = time.time()
    if now - fps_last >= 1.0:
        fps_text = fps_counter / (now - fps_last)
        fps_counter = 0
        fps_last = now
    avg_infer = infer_time / max(infer_count, 1) * 1000
    cv2.putText(display, f"FPS:{fps_text:.0f} infer:{avg_infer:.0f}ms", (5, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Track idle time — if no inference for too long, exit
    if idle_start and time.time() - idle_start > MAX_IDLE:
        print(f"No successful inference for {MAX_IDLE:.0f}s. Exiting.")
        break

    # --- bar chart ---
    chart = np.ones((h, CHART_W, 3), dtype=np.uint8) * 25

    current = emotion_history[-1] if emotion_history else None
    cur_emotions = current.get("emotions", {}) if current else {}
    cur_dominant = current.get("dominant_emotion", "?") if current else "?"
    cur_conf = current.get("confidence", 0) if current else 0

    bar_top = 38
    bar_bot = h - 18
    gap = 3
    n_emos = len(EMOTION_ORDER)
    bar_h = (bar_bot - bar_top - gap * (n_emos - 1)) // n_emos
    bar_left = 65
    bar_right = CHART_W - 8
    bar_w = bar_right - bar_left

    cv2.putText(chart, "Live Emotions", (bar_left, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    for pct in [0, 50, 100]:
        px = bar_left + int(pct / 100 * bar_w)
        cv2.line(chart, (px, bar_top), (px, bar_bot + gap * (n_emos - 1) + bar_h), (45, 45, 45), 1)

    for i, emo in enumerate(EMOTION_ORDER):
        y = bar_top + i * (bar_h + gap)
        col = EMOTION_COLORS.get(emo, (255, 255, 255))

        cv2.putText(chart, emo[:4], (2, y + bar_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
        cv2.rectangle(chart, (bar_left, y), (bar_right, y + bar_h), (48, 48, 48), -1)

        val = max(0.0, min(100.0, cur_emotions.get(emo, 0))) / 100.0
        fw = int(val * bar_w)
        if fw > 0:
            cv2.rectangle(chart, (bar_left, y), (bar_left + fw, y + bar_h), col, -1)
        cv2.putText(chart, f"{val*100:.0f}%", (bar_left + fw + 3, y + bar_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)

    dom_color = EMOTION_COLORS.get(cur_dominant, (255, 255, 255))
    cv2.putText(chart, f"{cur_dominant}  {cur_conf:.0%}", (bar_left, h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, dom_color, 1)

    combined = np.hstack([display, chart])
    cv2.imshow("Face Emotion", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        fname = os.path.join(csv_dir, f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(fname, combined)
        print(f"Saved: {fname}")

CAP.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"Session saved to: {csv_path}")

# Auto-generate HTML report
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from emotion_report import build_report
html = build_report(csv_path)
if html:
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"report_{basename}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to: {report_path}")
