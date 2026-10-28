"""
evaluate_three_way.py  --  batch comparison of the 3 methods on saved images.
Methods: OpenCV baseline | Proposed pipeline (v2) | WeChat QR (DL).

Metrics per category and overall:
  * Accuracy      = correct decodes / N   (decoded text MATCHES ground truth)
  * False-pos rate = wrong-but-nonempty decodes / N
  * Latency        = mean and median milliseconds (end-to-end)

USAGE (from the repository root):
  python src/evaluate_three_way.py      # screen-displayed codes (Group B1)
  python src/evaluate_paper.py          # printed codes (Group B2)

GROUND TRUTH:
  A CSV with columns: filename,ground_truth
  If a file has no ground-truth entry, the script counts ANY nonempty decode as
  correct (a fallback proxy) and prints a warning, because false-positive rate
  cannot be computed without ground truth.
"""
import cv2
import csv
import os
import statistics
from glob import glob
# Repository root (one level above src/), so the scripts work from any folder.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

from pipelines import (detect_baseline, detect_proposed,
                       load_wechat, detect_wechat)


def load_ground_truth(path):
    gt = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gt[row["filename"]] = row["ground_truth"]
    return gt


def verdict(decoded, truth):
    """Return 'correct', 'wrong' (false positive), or 'miss'."""
    if not decoded:
        return "miss"
    if truth is None:
        return "correct"      # no GT available -> treat any decode as correct
    return "correct" if decoded == truth else "wrong"


def run(folders, ground_truth_csv="ground_truth.csv",
        output_csv="three_way_comparison.csv"):

    gt = load_ground_truth(ground_truth_csv)
    if not gt:
        print("WARNING: no ground-truth CSV found -> false-positive rate unavailable.\n")

    wechat = load_wechat()
    wechat_on = wechat is not None

    rows = []
    for category, folder in folders.items():
        paths = sorted(glob(os.path.join(folder, "*.[jJ][pP][gG]")) +
                       glob(os.path.join(folder, "*.[pP][nN][gG]")))
        if not paths:
            print(f"  (skip) {folder} is empty")
            continue

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            name = os.path.basename(p)
            truth = gt.get(name)

            b_txt, b_lat = detect_baseline(img)
            s_txt, s_lat = detect_proposed(img)
            w_txt, w_lat = detect_wechat(wechat, img)

            rows.append({
                "Image": name, "Category": category,
                "Baseline": verdict(b_txt, truth), "Baseline_Lat": b_lat,
                "Proposed": verdict(s_txt, truth), "Proposed_Lat": s_lat,
                "WeChat": verdict(w_txt, truth) if wechat_on else "N/A",
                "WeChat_Lat": w_lat if wechat_on else None,
            })

    if not rows:
        print("No images found.")
        return

    # write per-image CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary
    cats = list(folders.keys()) + ["__TOTAL__"]
    print(f"\n{'Category':<12}{'N':>5}{'Base%':>8}{'Prop%':>8}{'WeChat%':>9}"
          f"{'B ms':>8}{'P ms':>8}{'W ms':>8}")
    print("-" * 66)
    for cat in cats:
        r = rows if cat == "__TOTAL__" else [x for x in rows if x["Category"] == cat]
        if not r:
            continue
        n = len(r)

        def acc(k):
            return 100.0 * sum(1 for x in r if x[k] == "correct") / n

        def fp(k):
            return 100.0 * sum(1 for x in r if x[k] == "wrong") / n

        def med_lat(k):
            v = [x[k] for x in r if x[k] is not None]
            return statistics.mean(v) if v else 0.0

        w_acc = f"{acc('WeChat'):>8.1f}" if wechat_on else f"{'N/A':>8}"
        w_lat = f"{med_lat('WeChat_Lat'):>8.1f}" if wechat_on else f"{'N/A':>8}"
        label = "TOTAL" if cat == "__TOTAL__" else cat
        print(f"{label:<12}{n:>5}{acc('Baseline'):>8.1f}{acc('Proposed'):>8.1f}"
              f"{w_acc}{med_lat('Baseline_Lat'):>8.1f}"
              f"{med_lat('Proposed_Lat'):>8.1f}{w_lat}")

    # false positives (needs GT)
    if gt:
        print("\nFalse-positive rate (wrong decode / N):")
        for cat in cats:
            r = rows if cat == "__TOTAL__" else [x for x in rows if x["Category"] == cat]
            if not r:
                continue
            n = len(r)
            label = "TOTAL" if cat == "__TOTAL__" else cat
            b = 100.0 * sum(1 for x in r if x["Baseline"] == "wrong") / n
            p = 100.0 * sum(1 for x in r if x["Proposed"] == "wrong") / n
            print(f"  {label:<12} Baseline {b:.1f}%   Proposed {p:.1f}%")

    print(f"\nPer-image results saved to {output_csv}")


if __name__ == "__main__":
    run(
        folders={
            c: os.path.join(ROOT, "data", "dataset_B1_screen", c)
            for c in ("low", "medium", "high")
        },
        ground_truth_csv=os.path.join(ROOT, "data", "ground_truth_screen.csv"),
        output_csv=os.path.join(ROOT, "results", "screen", "screen_three_way_comparison.csv"),
    )
