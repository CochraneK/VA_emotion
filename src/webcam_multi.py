r"""
Real-time multi-face webcam emotion detection.

Uses InsightFace for multi-face detection (no max limit), then runs
a lightweight heuristic on each face ROI. Displays each face's emotion
with colored bounding boxes and per-face labels.

CSV logging supports multiple faces per frame — each face gets its own
row tagged with a face_id (assigned by left-to-right order).

Usage:
    python webcam_multi.py                    # InsightFace multi-face
    python webcam_multi.py --label couple     # tag the session

Output: output/csv/emotions_<label>_<timestamp>.csv
"""
import sys
import os

# Suppress onnxruntime CUDA warnings BEFORE importing anything else
os.environ['ORT_LOGGING_LEVEL'] = '1'
os.environ['ONNXRUNTIME_LOG_SEVERITY'] = '3'

import time
import csv
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# --- Redirect stderr to suppress InsightFace/onnxruntime verbose output ---
# onnxruntime writes directly to the C runtime stderr, so we need to close
# the underlying file descriptor to truly silence it.
_original_stderr_fd = 2  # FD of stderr on Windows is 2
try:
    os.close(_original_stderr_fd)
except OSError:
    pass

class _DevNull:
    def write(self, s): pass
    def flush(self): pass

_original_stderr = sys.stderr
sys.stderr = _DevNull()

import warnings
warnings.filterwarnings('ignore')

# Set onnxruntime environment variables BEFORE any C-level import
os.environ['ORT_LOGGING_LEVEL'] = '1'       # 1 = Verbose (lowest priority)
os.environ['ONNXRUNTIME_VERBOSITY'] = '0'

sys.path.insert(0, os.path.dirname(__file__))
from backend_insightface import analyze

# Restore stderr so our prints show up (stdout is already captured by console)
sys.stderr = _original_stderr
print("webcam_multi: starting...")

# ---- Config ------------------------------------------------------ #
SKIP = 3            # analyze every N frames (InsightFace is fast on CPU)
CAM_W = 640         # capture width (wider for multi-face scenes)
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
# Color palette for distinguishing face IDs on screen
FACE_BOX_COLORS = [
    (0, 255, 0),     # green
    (255, 165, 0),   # orange
    (0, 191, 255),   # deep sky blue
    (255, 0, 255),   # magenta
    (255, 255, 0),   # yellow
    (139, 0, 255),   # purple
]
EMOTION_ORDER = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

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

csv_path = os.path.join(csv_dir, f"emotions{label_part}_{ts}.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["elapsed_s", "wall_time", "frame_idx", "face_id", "dominant_emotion"] + EMOTION_ORDER + ["n_faces"])
print(f"Logging to: {csv_path}")

# --- webcam ---
def open_camera():
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            print(f"  Camera {idx}: cannot open")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAM_W * 3 / 4))
        # DShow needs 2-3 warmup frames
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret and frame is not None and frame.mean() > 5:
            print(f"  Camera {idx}: open OK, brightness={frame.mean():.1f}")
            return cap, frame
        print(f"  Camera {idx}: warmup failed (ret={ret}, brightness={frame.mean() if frame is not None else 'N/A'})")
        cap.release()
        time.sleep(0.3)
    return None, None

CAP, first_frame = open_camera()
if CAP is None:
    csv_file.close()
    sys.exit("Error: cannot open webcam")

H, W = first_frame.shape[:2]

# Chart area: right side of the display
CHART_W = max(320, min(480, int(W * 0.35)))

frame_idx = 0
last_result = []     # list of face results for current frame
fps_text = 0.0
fps_counter = 0
fps_last = time.time()
start_time = time.time()
infer_time = 0.0
infer_count = 0
bad_read_count = 0
MAX_IDLE = 30.0
idle_start = 0.0

print("Controls: 'q'=quit | 's'=screenshot")
print("Multi-face mode: InsightFace backend, up to 6 faces visible")

cv2.namedWindow("Multi-Face Emotion", cv2.WINDOW_NORMAL)
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
        idle_start = time.time()

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
            last_result = []
        elif isinstance(raw, list):
            # Sort by x1 (left-to-right) for stable face_id assignment
            last_result = sorted(raw, key=lambda r: r["bbox"][0])
        else:
            last_result = [raw]

        if frame_idx % 60 == 0:
            print(f"  frame {frame_idx} | faces: {len(last_result)} | infer: {dt*1000:.0f}ms")

        # CSV: one row per face
        for fid, entry in enumerate(last_result):
            emotions = entry.get("emotions", {})
            row = [round(elapsed, 3), datetime.now().isoformat(timespec="milliseconds"),
                   frame_idx, fid, entry.get("dominant_emotion", "?")]
            for e in EMOTION_ORDER:
                row.append(round(emotions.get(e, 0), 1))
            row.append(len(last_result))
            csv_writer.writerow(row)
        csv_file.flush()

    # --- face overlay ---
    display = frame.copy()
    if last_result:
        for fid, r in enumerate(last_result):
            emotion = r.get("dominant_emotion", "?")
            conf = r.get("confidence", 0)
            label = f"F{fid}: {emotion} ({conf:.1%})"
            color = FACE_BOX_COLORS[fid % len(FACE_BOX_COLORS)]
            if "bbox" in r:
                bx = [int(r["bbox"][i]) for i in range(4)]
                cv2.rectangle(display, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                # Label above the box
                txt_h = 16
                cv2.rectangle(display, (bx[0], max(0, bx[1] - txt_h - 6)),
                              (bx[0] + 100, bx[1]), color, -1)
                cv2.putText(display, label, (bx[0] + 2, bx[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    if session_label:
        cv2.putText(display, f"[{session_label}] Faces: {len(last_result)}",
                    (w - 280, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

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

    # Track idle time
    if idle_start and time.time() - idle_start > MAX_IDLE:
        print(f"No successful inference for {MAX_IDLE:.0f}s. Exiting.")
        break

    # --- bar chart (aggregate across all faces, average) ---
    chart = np.ones((H, CHART_W, 3), dtype=np.uint8) * 25

    if last_result:
        # Average emotions across all faces
        agg = {e: 0.0 for e in EMOTION_ORDER}
        for r in last_result:
            em = r.get("emotions", {})
            for e in EMOTION_ORDER:
                agg[e] += em.get(e, 0)
        for e in agg:
            agg[e] /= len(last_result)
    else:
        agg = {e: 0.0 for e in EMOTION_ORDER}

    bar_top = 38
    bar_bot = H - 18
    gap = 3
    n_emos = len(EMOTION_ORDER)
    bar_h = (bar_bot - bar_top - gap * (n_emos - 1)) // n_emos
    bar_left = 65
    bar_right = CHART_W - 8
    bar_w = bar_right - bar_left

    cv2.putText(chart, f"Live Emotions (n={len(last_result)})", (bar_left, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    for pct in [0, 50, 100]:
        px = bar_left + int(pct / 100 * bar_w)
        cv2.line(chart, (px, bar_top), (px, bar_bot + gap * (n_emos - 1) + bar_h), (45, 45, 45), 1)

    for i, emo in enumerate(EMOTION_ORDER):
        y = bar_top + i * (bar_h + gap)
        col = EMOTION_COLORS.get(emo, (255, 255, 255))

        cv2.putText(chart, emo[:4], (2, y + bar_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
        cv2.rectangle(chart, (bar_left, y), (bar_right, y + bar_h), (48, 48, 48), -1)

        val = max(0.0, min(100.0, agg.get(emo, 0))) / 100.0
        fw = int(val * bar_w)
        if fw > 0:
            cv2.rectangle(chart, (bar_left, y), (bar_left + fw, y + bar_h), col, -1)
        cv2.putText(chart, f"{val*100:.0f}%", (bar_left + fw + 3, y + bar_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)

    combined = np.hstack([display, chart])
    cv2.imshow("Multi-Face Emotion", combined)

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
