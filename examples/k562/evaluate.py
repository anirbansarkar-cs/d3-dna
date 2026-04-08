"""
Evaluate generated K562 sequences against reference test data.

Built-in SP-MSE evaluation (no external dependencies beyond mpralegnet oracle):
    python evaluate.py --data data/lenti_MPRA_K562_data.h5 --oracle path/to/oracle.ckpt

Full evaluation with external pipeline (5 metrics):
    python evaluate.py --data data/lenti_MPRA_K562_data.h5 --oracle path/to/oracle.ckpt \
        --eval-pipeline /path/to/d3_evaluation_pipeline
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

import numpy as np
import torch

# Add mpralegnet to path for oracle loading
sys.path.insert(0, os.path.expanduser("~/mpralegnet"))

parser = argparse.ArgumentParser()
parser.add_argument("--samples-dir", type=str, default="generated",
                    help="Directory containing sample_*.npz files")
parser.add_argument("--data", type=str, required=True,
                    help="Path to K562 H5 data file")
parser.add_argument("--oracle", type=str, required=True,
                    help="Path to LegNet oracle checkpoint")
parser.add_argument("--eval-pipeline", type=str, default=None,
                    help="Path to d3_evaluation_pipeline directory (optional, for full 5-metric eval)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="Batch size for oracle inference")
parser.add_argument("--output-dir", type=str, default="eval_results",
                    help="Directory for evaluation results")
args = parser.parse_args()

assert os.path.isdir(args.samples_dir), f"Samples directory not found: {args.samples_dir}"

# Discover sample files
sample_files = sorted(glob.glob(os.path.join(args.samples_dir, "sample_*.npz")))
assert sample_files, f"No sample_*.npz files found in {args.samples_dir}"
print(f"Found {len(sample_files)} sample replicates")

os.makedirs(args.output_dir, exist_ok=True)

# ── Built-in SP-MSE evaluation ──────────────────────────────────────────────

print("\n=== SP-MSE Evaluation (built-in) ===")

# Load test data
import h5py
with h5py.File(args.data, "r") as f:
    x_test_onehot = np.array(f["onehot_test"])  # (N, 230, 4)

# (N, 230, 4) -> (N, 4, 230) for oracle input
x_test = torch.from_numpy(x_test_onehot).permute(0, 2, 1).float()

# Load oracle
from mpralegnet import LitModel
oracle = LitModel.load_from_checkpoint(args.oracle).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
oracle = oracle.to(device)

def oracle_predict(x_onehot, batch_size=256):
    """Run oracle predictions in batches. Input: (N, 4, seq_len) tensor."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_onehot), batch_size):
            batch = x_onehot[i:i + batch_size].to(device)
            pred = oracle.model(batch)
            preds.append(pred.cpu())
    return torch.cat(preds, dim=0)

# Oracle predictions on test data (shared across replicates)
print("Predicting on test sequences...")
y_test = oracle_predict(x_test, args.batch_size)

# Compute SP-MSE per replicate
sp_mse_results = {}
for npz_path in sample_files:
    name = os.path.splitext(os.path.basename(npz_path))[0]
    data = np.load(npz_path)
    x_gen_onehot = data["arr_0"]  # (N, 230, 4)
    x_gen = torch.from_numpy(x_gen_onehot).permute(0, 2, 1).float()

    y_gen = oracle_predict(x_gen, args.batch_size)
    mse = torch.mean((y_test - y_gen) ** 2).item()
    sp_mse_results[name] = mse
    print(f"  {name}: SP-MSE = {mse:.6f}")

# Save SP-MSE CSV
csv_path = os.path.join(args.output_dir, "sp_mse.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    names = sorted(sp_mse_results.keys())
    writer.writerow([""] + names)
    writer.writerow(["sp_mse"] + [sp_mse_results[n] for n in names])
print(f"\nSP-MSE results saved to {csv_path}")

mean_mse = np.mean(list(sp_mse_results.values()))
print(f"Mean SP-MSE across {len(sp_mse_results)} replicates: {mean_mse:.6f}")

# ── External pipeline (optional, for full 5-metric eval) ────────────────────

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
