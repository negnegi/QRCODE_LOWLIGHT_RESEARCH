"""
pipelines.py  --  the three detection methods, ONE source of truth.
Both the live capture tool and the batch evaluator import from here.

Method 2 (detect_proposed) reproduces EXACTLY the validated Smart Pipeline v2
from the Phase-1 experiments:
  * Strategy 1 - single-pass brightness-aware routing (mean grey level of the
    full-resolution greyscale frame):
        mean > 100          -> raw decode (no enhancement)
        40 < mean <= 100    -> CLAHE -> decode
        mean <= 40          -> gamma (gamma = 0.5) + CLAHE -> decode
  * Strategy 2 - after enhancement, downscale so the longer side is 320 px,
    then decode.
  * Exactly ONE decode attempt per image.
  * NO Gaussian adaptive thresholding stage (tested and dropped in the ablation
    because it lowers accuracy with cv2.QRCodeDetector, which binarises internally).
  * Latency is measured END-TO-END (preprocessing INCLUDED).

Validated Phase-1 result at 320 px (100-image feasibility set, i7-7500U):
  Normal 100.0% | Low-light 88.2% | Noisy 97.0% | Overall 95.0% | 16.8 ms avg.
"""
import cv2
import numpy as np
import time

# ---- validated v2 constants ----
GAMMA = 0.5
GAMMA_TABLE = np.array([((i / 255.0) ** GAMMA) * 255 for i in range(256)]).astype("uint8")
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
DOWNSCALE = 320
DETECTOR = cv2.QRCodeDetector()


def _downscale(img, target=DOWNSCALE):
    h, w = img.shape[:2]
    m = max(h, w)
    if m > target:
        s = target / m
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ---------- Method 1: OpenCV baseline (standard pipeline) ----------
def detect_baseline(img):
    """Returns (decoded_text, latency_ms). '' means no decode."""
    t0 = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data, _, _ = DETECTOR.detectAndDecode(gray)
    lat = (time.perf_counter() - t0) * 1000.0
    return data, lat


# ---------- Method 2: Proposed pipeline (validated Smart v2) ----------
def detect_proposed(img):
    """Returns (decoded_text, latency_ms). End-to-end timing.
    Faithful reproduction of the validated Phase-1 v2 pipeline."""
    t0 = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))                        # Strategy 1: routing

    if mean > 100:
        proc = img                                     # raw path (original image)
    elif mean > 40:
        proc = cv2.cvtColor(CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)      # CLAHE path
    else:
        corrected = cv2.LUT(gray, GAMMA_TABLE)         # gamma(0.5) + CLAHE path
        proc = cv2.cvtColor(CLAHE.apply(corrected), cv2.COLOR_GRAY2BGR)

    proc = _downscale(proc)                            # Strategy 2: 320 px
    data, _, _ = DETECTOR.detectAndDecode(proc)        # single decode
    lat = (time.perf_counter() - t0) * 1000.0
    return data, lat


# ---------- Method 3: WeChat QR (deep-learning reference) ----------
def load_wechat():
    """Load WeChat model files if present, else return None.
    The 4 files are expected in the repository's models/ folder. They come from
    https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode
    (detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel)."""
    import os
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    candidates = [
        tuple(os.path.join(model_dir, f) for f in
              ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")),
    ]
    for d, dc, s, sc in candidates:
        if all(os.path.exists(f) for f in (d, dc, s, sc)):
            try:
                return cv2.wechat_qrcode_WeChatQRCode(d, dc, s, sc)
            except Exception as e:
                print("WeChat load error:", e)
                return None
    print("WeChat model files not found - WeChat column will be N/A.")
    return None


def detect_wechat(detector, img):
    """Returns (decoded_text or None, latency_ms). None = model not loaded."""
    if detector is None:
        return None, 0.0
    t0 = time.perf_counter()
    res, _ = detector.detectAndDecode(img)
    lat = (time.perf_counter() - t0) * 1000.0
    return (res[0] if res else ""), lat
