"""
Unified face emotion analysis runner.
Supports: deepface (optional, needs tensorflow), mediapipe, insightface
Usage:
    python run_all.py --video video.mp4 --backend deepface
    python run_all.py --video video.mp4 --backend mediapipe
    python run_all.py --video video.mp4 --backend insightface
    python run_all.py --video video.mp4 --backend all --interval 15
    python compare.py --video video.mp4 --output results.json
"""
import argparse
import cv2
import json
import time
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SRC_DIR, "..", "output")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
sys.path.insert(0, SRC_DIR)
os.makedirs(CSV_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Face emotion analysis runner")
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument(
        "--backend",
        choices=["deepface", "mediapipe", "insightface", "all"],
        default="mediapipe",
        help="Which backend to use",
    )
    p.add_argument("--model", default="VGG-Face", help="Model name for deepface")
    p.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Frame skip interval (every N frames to analyze)",
    )
    p.add_argument("--output", default=None, help="Output JSON file (default: output/<video>_<backend>.json)")
    p.add_argument(
        "--draw",
        action="store_true",
        help="Draw emotion labels on output video",
    )
    return p.parse_args()


def run_backend(backend, frame, model_name="VGG-Face"):
    """Dispatch to the correct backend. Returns dict or list of dicts."""
    if backend == "deepface":
        from backend_deepface import analyze
        return analyze(frame, model_name=model_name)
    elif backend == "mediapipe":
        from backend_mediapipe import analyze
        return analyze(frame)
    elif backend == "insightface":
        from backend_insightface import analyze
        return analyze(frame)
    return None


def flatten_result(result):
    """Handle list or single dict result from backends."""
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def process_video(video_path, backend, model_name, interval, draw):
    """Process a video file frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")
    print(f"Backend: {backend}, Model: {model_name}")
    print(f"Analyze every {interval} frames")
    print()

    results = []
    frame_idx = 0
    analyze_idx = 0
    start_time = time.time()

    out = None
    if draw:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, "output_" + backend + ".avi"),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (w, h),
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % interval != 0:
            if draw and out:
                out.write(frame)
            continue

        analyze_idx += 1
        raw_result = run_backend(backend, frame, model_name)
        flat_results = flatten_result(raw_result)

        for r in flat_results:
            entry = {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 2),
                **r,
            }
            results.append(entry)

            if draw:
                emotion = r.get("dominant_emotion", "?")
                conf = r.get("confidence", 0)
                label = f"{emotion} ({conf:.1%})"
                if "bbox" in r:
                    bx = [int(r["bbox"][i]) for i in range(4)]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (bx[0], bx[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if analyze_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"\rProgress: {analyze_idx} frames analyzed ({elapsed:.1f}s)", end="", flush=True)

        if draw and out:
            out.write(frame)

    cap.release()
    if out:
        out.release()

    elapsed = time.time() - start_time
    faces_found = len(results)
    print(f"\rComplete! {analyze_idx} frames analyzed, {faces_found} face(s) found ({elapsed:.1f}s)")
    print()
    return results


def print_summary(results):
    """Print emotion distribution summary."""
    if not results:
        print("No faces detected.")
        return
    emotions = {}
    for r in results:
        e = r.get("dominant_emotion", "?")
        emotions[e] = emotions.get(e, 0) + 1
    total = len(results)
    print(f"\nEmotion distribution ({total} faces):")
    for e, c in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"  {e}: {c} frames ({c/total:.0%})")


def main():
    args = parse_args()

    # Show backend availability
    print("Backend availability:")
    from backend_deepface import DEEPFACE_AVAILABLE
    print(f"  deepface: {'available' if DEEPFACE_AVAILABLE else 'unavailable (needs tensorflow)'}")
    print(f"  mediapipe: available")
    print(f"  insightface: available")
    print()

    if args.output is None:
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(CSV_DIR, f"{video_stem}_{args.backend}.json")

    if args.backend == "all":
        all_results = {}
        for bk in ["deepface", "mediapipe", "insightface"]:
            if bk == "deepface" and not DEEPFACE_AVAILABLE:
                print(f"Skipping deepface (not available)\n")
                continue
            print(f"\n{'='*50}")
            print(f"Running backend: {bk}")
            print(f"{'='*50}")
            model = "VGG-Face" if bk == "deepface" else "auto"
            res = process_video(args.video, bk, model, args.interval, args.draw)
            if res:
                all_results[bk] = res
            args.draw = False

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nAll results saved to {args.output}")

        for bk, res in all_results.items():
            print(f"\n--- {bk} ---")
            print_summary(res)

    else:
        results = process_video(args.video, args.backend, args.model, args.interval, args.draw)
        if results:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")
        print_summary(results)


if __name__ == "__main__":
    main()
