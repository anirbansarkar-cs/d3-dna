"""
Evaluate generated K562 sequences against reference test data.

Uses the vendored LegNet oracle in ``examples/k562/oracle.py`` (no external
mpralegnet dependency). For each ``sample_*.npz`` replicate, computes SP-MSE
against the test-set oracle predictions.

Usage (built-in SP-MSE over all replicates):
    python evaluate.py --data data/lenti_MPRA_K562_data.h5 --oracle path/to/oracle.ckpt

Optional external 5-metric pipeline:
    python evaluate.py --data data/lenti_MPRA_K562_data.h5 --oracle path/to/oracle.ckpt \
        --eval-pipeline /path/to/d3_evaluation_pipeline
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

import h5py
import numpy as np
import torch

from oracle import load as load_legnet_oracle


parser = argparse.ArgumentParser()
parser.add_argument("--samples-dir", type=str, default="generated",
                    help="Directory containing sample_*.npz files")
parser.add_argument("--data", type=str, required=True,
                    help="Path to K562 H5 data file")
parser.add_argument("--oracle", type=str, required=True,
                    help="Path to LegNet oracle checkpoint")
parser.add_argument("--oracle-config", type=str, default=None,
                    help="Optional LegNet config.json path (defaults to oracle.DEFAULT_CONFIG)")
parser.add_argument("--eval-pipeline", type=str, default=None,
                    help="Path to d3_evaluation_pipeline directory (optional, for full 5-metric eval)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="Batch size for oracle inference")
parser.add_argument("--output-dir", type=str, default="eval_results",
                    help="Directory for evaluation results")
args = parser.parse_args()

assert os.path.isdir(args.samples_dir), f"Samples directory not found: {args.samples_dir}"

sample_files = sorted(glob.glob(os.path.join(args.samples_dir, "sample_*.npz")))
assert sample_files, f"No sample_*.npz files found in {args.samples_dir}"
print(f"Found {len(sample_files)} sample replicates")

os.makedirs(args.output_dir, exist_ok=True)

# ── Built-in SP-MSE evaluation ──────────────────────────────────────────────

print("\n=== SP-MSE Evaluation (built-in) ===")

with h5py.File(args.data, "r") as f:
    x_test_onehot = np.array(f["onehot_test"])  # (N, 230, 4)

# (N, 230, 4) -> (N, 4, 230) for oracle input
x_test = np.transpose(x_test_onehot, (0, 2, 1)).astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
oracle = load_legnet_oracle(args.oracle, device, config_path=args.oracle_config)

print("Predicting on test sequences...")
y_test = oracle.predict(x_test, batch_size=args.batch_size)  # (N, 1)

sp_mse_results = {}
for npz_path in sample_files:
    name = os.path.splitext(os.path.basename(npz_path))[0]
    data = np.load(npz_path)
    x_gen_onehot = data["arr_0"]  # (N, 230, 4)
    x_gen = np.transpose(x_gen_onehot, (0, 2, 1)).astype(np.float32)

    y_gen = oracle.predict(x_gen, batch_size=args.batch_size)  # (N, 1)
    mse = float(np.mean((y_test - y_gen) ** 2))
    sp_mse_results[name] = mse
    print(f"  {name}: SP-MSE = {mse:.6f}")

csv_path = os.path.join(args.output_dir, "sp_mse.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    names = sorted(sp_mse_results.keys())
    writer.writerow([""] + names)
    writer.writerow(["sp_mse"] + [sp_mse_results[n] for n in names])
print(f"\nSP-MSE results saved to {csv_path}")

mean_mse = float(np.mean(list(sp_mse_results.values())))
print(f"Mean SP-MSE across {len(sp_mse_results)} replicates: {mean_mse:.6f}")

# ── External pipeline (optional) ────────────────────────────────────────────

if args.eval_pipeline:
    print(f"\n=== External Pipeline Evaluation (5 metrics) ===")
    assert os.path.isdir(args.eval_pipeline), f"Evaluation pipeline not found: {args.eval_pipeline}"

    cmd = [
        sys.executable, os.path.join(args.eval_pipeline, "main.py"),
        "--samples-batch", args.samples_dir,
        "--data", args.data,
        "--model", args.oracle,
        "--model-type", "lentimpra",
        "--test", "discriminability,predictive_dist_shift,kmer_spectrum_shift,cond_gen_fidelity,percent_identity",
        "--output-dir", args.output_dir,
    ]

    print(f"Running: {' '.join(cmd)}\n")

    metadata_csv = os.path.join(args.samples_dir, "metadata.csv")
    if not os.path.exists(metadata_csv):
        print("First run: creating metadata.csv template...")
        subprocess.run(cmd, check=False)

    print("Running evaluation...")
    subprocess.run(cmd, check=True)

# ── Print summary ───────────────────────────────────────────────────────────

print("\n=== Results ===")
for csv_file in sorted(os.listdir(args.output_dir)):
    if csv_file.endswith(".csv"):
        path = os.path.join(args.output_dir, csv_file)
        print(f"\n{csv_file}:")
        with open(path) as f:
            for line in f:
                print(f"  {line.rstrip()}")
