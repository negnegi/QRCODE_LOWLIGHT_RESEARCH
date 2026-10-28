"""
LIVE CAPTURE + DECODE  (run this on YOUR laptop, not in the sandbox)
--------------------------------------------------------------------
What it does:
  1. Connects to your laptop camera.
  2. Shows the live picture on screen.
  3. Runs the QR detector on every frame and measures the time (latency).
  4. When you press 's', it SAVES the current frame and writes a row to results.csv
     (filename, condition, decoded text, latency, whether it matched the ground truth).
  5. Press 'q' to quit.

HOW TO RUN (from the repository root):
  pip install -r requirements.txt
  python src/capture_and_decode.py --condition low

Change --condition to "medium" or "high" for the other batches.

The live preview uses the proposed pipeline (detect_proposed in pipelines.py).
"""
import cv2
import time
import csv
import os
import argparse

# ----------------------------------------------------------------------
# 1. HOW THE ALGORITHM CONNECTS TO THE CAMERA
# ----------------------------------------------------------------------
# cv2.VideoCapture(0) opens the DEFAULT laptop camera.
#   0 = built-in webcam (most laptops)
#   1 or 2 = an external USB camera, if you have one
# That single line IS the connection between your algorithm and the camera.
CAMERA_INDEX = 0

# ----------------------------------------------------------------------
# 2. THE DETECTOR  (shared with the batch evaluator via pipelines.py)
# ----------------------------------------------------------------------
# We import the SAME proposed pipeline used in evaluate_three_way.py, so the
# live capture and the batch comparison always run identical code.
from pipelines import detect_proposed

def detect_and_decode(frame):
    """Return the decoded string ('' if nothing found)."""
    value, _latency_ms = detect_proposed(frame)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="low",
                        choices=["low", "medium", "high"],
                        help="Lighting condition of this batch")
    parser.add_argument("--outdir", default="captures",
                        help="Where to save captured images")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results_path = "results.csv"
    new_file = not os.path.exists(results_path)
    results_file = open(results_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(results_file)
    if new_file:
        writer.writerow(["saved_filename", "condition", "decoded_text",
                         "latency_ms", "timestamp"])

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: could not open the camera. Try CAMERA_INDEX = 1.")
        return

    print("Camera open. Show a QR code to the camera.")
    print("Press 's' to SAVE the current frame, 'q' to QUIT.")

    saved_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: could not read a frame.")
            break

        # Measure how long the algorithm takes on this frame (latency).
        start = time.perf_counter()
        decoded = detect_and_decode(frame)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Draw the result on the live picture so you can see it.
        status = f"[{args.condition}]  decoded: {decoded if decoded else '---'}   {latency_ms:.1f} ms"
        color = (0, 200, 0) if decoded else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        cv2.putText(frame, "s = save    q = quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("QR capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            saved_count += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{args.condition}_{saved_count:04d}_{ts}.png"
            cv2.imwrite(os.path.join(args.outdir, fname), frame)
            writer.writerow([fname, args.condition, decoded,
                             f"{latency_ms:.1f}", ts])
            results_file.flush()
            print(f"saved {fname}  decoded={decoded!r}  {latency_ms:.1f} ms")

    cap.release()
    cv2.destroyAllWindows()
    results_file.close()
    print(f"Done. Saved {saved_count} images. Results in results.csv")


if __name__ == "__main__":
    main()
