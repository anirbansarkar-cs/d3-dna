"""
Evaluate generated K562 sequences against reference test data.

Runs the d3_evaluation_pipeline to compute:
  - Discriminability (AUROC)
  - Predictive distribution shift (KS statistic)
  - K-mer spectrum shift (JS distance)
  - Conditional generation fidelity (MSE)

Usage:
    python evaluate.py
    python evaluate.py --samples-dir generated --output-dir eval_results
"""

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--samples-dir", type=str, default="generated",
                    help="Directory containing sample_*.npz files")
parser.add_argument("--data", type=str, required=True,
                    help="Path to K562 H5 data file")
parser.add_argument("--oracle", type=str, required=True,
                    help="Path to LegNet oracle checkpoint")
parser.add_argument("--eval-pipeline", type=str, required=True,
                    help="Path to d3_evaluation_pipeline directory")
parser.add_argument("--output-dir", type=str, default="eval_results",
                    help="Directory for evaluation results")
args = parser.parse_args()

EVAL_PIPELINE = args.eval_pipeline
assert os.path.isdir(EVAL_PIPELINE), f"Evaluation pipeline not found: {EVAL_PIPELINE}"
assert os.path.isdir(args.samples_dir), f"Samples directory not found: {args.samples_dir}"

cmd = [
    sys.executable, os.path.join(EVAL_PIPELINE, "main.py"),
    "--samples-batch", args.samples_dir,
    "--data", args.data,
    "--model", args.oracle,
    "--model-type", "lentimpra",
    "--test", "discriminability,predictive_dist_shift,kmer_spectrum_shift,cond_gen_fidelity,percent_identity",
    "--output-dir", args.output_dir,
]

print(f"Running evaluation pipeline:\n  {' '.join(cmd)}\n")

# Batch mode creates metadata.csv on first run, then processes on second run
metadata_csv = os.path.join(args.samples_dir, "metadata.csv")
if not os.path.exists(metadata_csv):
    print("First run: creating metadata.csv template...")
    subprocess.run(cmd, check=False)

print("Running evaluation...")
subprocess.run(cmd, check=True)

# Print summary
print("\n=== Results ===")
if os.path.isdir(args.output_dir):
    for csv_file in sorted(os.listdir(args.output_dir)):
        if csv_file.endswith(".csv"):
            path = os.path.join(args.output_dir, csv_file)
            print(f"\n{csv_file}:")
            with open(path) as f:
                for line in f:
                    print(f"  {line.rstrip()}")
else:
    print(f"No results directory found: {args.output_dir}")
