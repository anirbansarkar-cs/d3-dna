"""Run SP-MSE on HepG2 samples via the d3-dna package evaluator.

Loads onehot_test from the HepG2 H5, loads the LegNet oracle via the vendored
examples/hepg2/oracle.py, and calls d3_dna.evals.compute_fidelity_mse for each
sample_*.npz replicate. Cross-checks the external d3_evaluation_pipeline's
cond_gen_fidelity_mse column.
"""

import argparse
import csv
import glob
import os

import h5py
import numpy as np
import torch

from d3_dna.evals import compute_fidelity_mse
from oracle import load as load_legnet_oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", required=True,
                        help="Directory containing sample_*.npz files")
    parser.add_argument("--data", required=True,
                        help="HepG2 H5 with onehot_test (N, 230, 4)")
    parser.add_argument("--oracle", required=True,
                        help="LegNet HepG2 oracle checkpoint")
    parser.add_argument("--oracle-config", default=None,
                        help="Optional LegNet config.json (defaults to oracle.DEFAULT_CONFIG)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    assert os.path.isdir(args.samples_dir), f"Samples dir not found: {args.samples_dir}"
    os.makedirs(args.output_dir, exist_ok=True)

    sample_files = sorted(glob.glob(os.path.join(args.samples_dir, "sample_*.npz")))
    assert sample_files, f"No sample_*.npz in {args.samples_dir}"
    print(f"Found {len(sample_files)} sample replicates")

    with h5py.File(args.data, "r") as f:
        x_test_onehot = np.array(f["onehot_test"])  # (N, 230, 4)
    x_test = np.transpose(x_test_onehot, (0, 2, 1)).astype(np.float32)  # (N, 4, 230)
    print(f"Loaded test set: {x_test.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle = load_legnet_oracle(args.oracle, device, config_path=args.oracle_config)

    print("Predicting oracle on test sequences...")
    y_test = oracle.predict(x_test, batch_size=args.batch_size)  # (N, 1)
    print(f"  y_test: {y_test.shape}")

    sp_mse_results = {}
    for npz_path in sample_files:
        name = os.path.splitext(os.path.basename(npz_path))[0]
        x_gen_onehot = np.load(npz_path)["arr_0"]  # (N, 230, 4)
        x_gen = np.transpose(x_gen_onehot, (0, 2, 1)).astype(np.float32)
        y_gen = oracle.predict(x_gen, batch_size=args.batch_size)
        mse = compute_fidelity_mse(y_test, y_gen)
        sp_mse_results[name] = mse
        print(f"  {name}: SP-MSE = {mse:.6f}")

    csv_path = os.path.join(args.output_dir, "sp_mse.csv")
    names = sorted(sp_mse_results.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        w.writerow(["sp_mse"] + [sp_mse_results[n] for n in names])
    print(f"\nSaved {csv_path}")
    print(f"Mean SP-MSE across {len(names)} replicates: {float(np.mean(list(sp_mse_results.values()))):.6f}")


if __name__ == "__main__":
    main()
