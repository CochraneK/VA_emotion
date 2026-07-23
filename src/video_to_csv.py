"""
Analyze a video file with HSEmotion (GPU) and output an emotion CSV.
Compatible with emotion_report.py and emotion_sync.py.
Usage:
    python video_to_csv.py --video path/to/video.mp4 [--label person_a] [--skip 10]
"""
import sys, os, time, csv, argparse
from datetime import datetime
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from backend_hsemotion import analyze

EMOTION_ORDER = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--skip", type=int, default=10, help="Analyze every N frames")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f} fps, analyzing every {args.skip} frames")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_part = f"_{args.label}" if args.label else ""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "csv")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"emotions{label_part}_{ts}.csv")
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["elapsed_s", "wall_time", "frame_idx", "dominant_emotion"] + EMOTION_ORDER + ["n_faces"])

    frame_idx = 0
    analyzed = 0
    faces_found = 0
    start = time.time()
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = frame_idx / max(fps, 1)
        frame_idx += 1

        if frame_idx % args.skip != 0:
            continue

        analyzed += 1
        result = analyze(frame)
        if result is None:
            if analyzed % 20 == 0:
                print(f"  frame {frame_idx}/{total} ({analyzed} analyzed, {faces_found} faces)...")
            continue

        faces_found += 1
        last_result = result
        emotions = result.get("emotions", {})
        row = [round(elapsed, 3), datetime.now().isoformat(timespec="milliseconds"),
               frame_idx, result.get("dominant_emotion", "?")]
        for e in EMOTION_ORDER:
            row.append(round(emotions.get(e, 0), 1))
        row.append(1)
        writer.writerow(row)

        if analyzed % 20 == 0:
            print(f"  frame {frame_idx}/{total} ({analyzed} analyzed, {faces_found} faces)...")

    cap.release()
    f.close()

    elapsed = time.time() - start
    print(f"\nDone! {analyzed} frames analyzed, {faces_found} faces found in {elapsed:.1f}s")
    print(f"CSV saved to: {csv_path}")

    # Auto-generate HTML report
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from emotion_report import build_report
    html = build_report(csv_path)
    if html:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        reports_dir = os.path.join(os.path.dirname(output_dir), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"report_{basename}.html")
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write(html)
        print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
