"""Quick diagnostic: open camera and test InsightFace face detection."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from backend_insightface import analyze

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera!")
    sys.exit(1)

# Warmup
for _ in range(5):
    cap.read()

ret, frame = cap.read()
if not ret:
    print("Failed to read frame!")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame: {w}x{h}, brightness={frame.mean():.1f}")

# Test InsightFace detection
from insightface.app import FaceAnalysis as FA
app = FA(name="buffalo_s", root=os.path.dirname(os.path.abspath(__file__)))
app.prepare(ctx_id=0, det_size=(640, 640))
bboxes = app.det_model.detect_faces(frame, det_thresh=0.5)
print(f"InsightFace detection: {len(bboxes)} face(s)")
if bboxes is None:
    print("  -> bboxes is None (detector failed)")
elif len(bboxes) == 0:
    print("  -> bboxes is empty list (no faces at threshold 0.5)")
else:
    for i, b in enumerate(bboxes):
        print(f"  face {i}: bbox={b[:4]}, conf={b[4]:.2f}")

# Try emotion analysis
result = analyze(frame)
print(f"analyze() returned: {result is not None}")
if result:
    print(f"  -> {len(result) if isinstance(result, list) else 'single'} face(s)")
    if isinstance(result, list):
        for r in result:
            print(f"  {r}")

cap.release()
print("\n--- Now showing camera (any key to quit) ---")
while True:
    ret, f = cap.read() if cap.isOpened() else (False, None)
    if not ret:
        break
    cv2.imshow("diag", f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
