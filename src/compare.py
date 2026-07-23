"""
Compare all backends on the same video and produce a side-by-side report.
Usage:
    python compare.py --video test.mp4 --output compare_results.json
"""
import argparse
import cv2
import json
import time
from collections import Counter

from run_all import process_video


def parse_args():
    p = argparse.ArgumentParser(description="Compare all backends side-by-side")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="compare_results.json")
    p.add_argument("--interval", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Comparing backends on: {args.video}")
    print()

    results = {}
    for bk in ["deepface", "mediapipe", "insightface"]:
        print(f"[{bk}/3] Running...")
        t0 = time.time()
        res = process_video(args.video, bk, "VGG-Face", args.interval, draw=False)
        if res:
            results[bk] = res
        elapsed = time.time() - t0
        faces = len(res) if res else 0
        print(f"[{bk}/3] Done: {faces} faces in {elapsed:.1f}s ({elapsed/max(faces,1):.2f}s/face)\n")

    # Aggregate emotion distributions per backend
    summary = {}
    for bk, res in results.items():
        emotions = Counter(r.get("dominant_emotion", "?") for r in res)
        total = sum(emotions.values())
        summary[bk] = {emotion: {"count": count, "pct": round(count / total, 3)}
                       for emotion, count in emotions.most_common()}

    report = {
        "video": args.video,
        "backend_results": results,
        "summary": summary,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Backend':<20} {'Total Frames':>12} {'Faces Found':>12} {'Time (s)':>10} {'Top Emotion':>12}")
    print("-" * 60)
    for bk, res in results.items():
        total_frames = len(res)
        emotions = Counter(r.get("dominant_emotion", "?") for r in res)
        top = emotions.most_common(1)[0] if emotions else ("?", 0)
        # Get timing from process_video output... we recompute
        elapsed_str = "?"
        print(f"{bk:<20} {total_frames:>12} {total_frames:>12} {elapsed_str:>10} {top[0]:>12}")

    print("\nReport saved to:", args.output)


if __name__ == "__main__":
    main()
