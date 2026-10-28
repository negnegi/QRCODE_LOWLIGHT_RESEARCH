"""
evaluate_paper.py  --  three-method comparison on the PRINTED-code dataset (Group B2).
Runs the OpenCV baseline, the proposed pipeline and WeChat QR on the SAME images
and reports accuracy (vs ground truth), false-positive rate and latency per method.

Usage (from the repository root):
    python src/evaluate_paper.py

Inputs:   data/dataset_B2_printed/{low,medium,high}/
          data/ground_truth_paper_ALL.csv
          models/  (the four WeChat model files)
Output:   results/paper/paper_three_way_comparison.csv
"""
import os
from evaluate_three_way import run, ROOT

if __name__ == "__main__":
    run(
        folders={
            c: os.path.join(ROOT, "data", "dataset_B2_printed", c)
            for c in ("low", "medium", "high")
        },
        ground_truth_csv=os.path.join(ROOT, "data", "ground_truth_paper_ALL.csv"),
        output_csv=os.path.join(ROOT, "results", "paper", "paper_three_way_comparison.csv"),
    )
