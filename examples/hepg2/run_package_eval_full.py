"""Run all 4 d3-dna package metrics on HepG2 samples.

Computes per-replicate: fidelity_mse, ks_statistic, js_spectrum (k=6), auroc.
Results saved to --output-dir as separate CSVs for comparison against
the external d3_evaluation_pipeline numbers.
"""

import argparse
import csv
import glob
import os

import h5py
import numpy as np
import torch

from d3_dna.evals import (
    compute_fidelity_mse,
    compute_ks_statistic,
    compute_js_spectrum,
    compute_auroc,
)
from oracle import load as load_legnet_oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", required=True,
                        help="Directory containing sample_*.npz files")
    parser.add_argument("--data", required=True,
                        help="HepG2 H5 with onehot_test (N, 230, 4)")
    parser.add_argument("--oracle", required=True,
                        help="LegNet HepG2 oracle checkpoint")
    parser.add_argument("--oracle-config", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tests", default="fidelity_mse,ks_statistic,js_k6,auroc",
                        help="Comma-separated subset of: fidelity_mse,ks_statistic,js_k6,auroc")
    args = parser.parse_args()

    ORACLE_TESTS = {"fidelity_mse", "ks_statistic"}
    tests = [t.strip() for t in args.tests.split(",")]

    assert os.path.isdir(args.samples_dir), f"Samples dir not found: {args.samples_dir}"
    os.makedirs(args.output_dir, exist_ok=True)

    sample_files = sorted(glob.glob(os.path.join(args.samples_dir, "sample_*.npz")))
    assert sample_files, f"No sample_*.npz in {args.samples_dir}"
    print(f"Found {len(sample_files)} sample replicates | tests: {tests}")

    with h5py.File(args.data, "r") as f:
        x_test_onehot = np.array(f["onehot_test"])  # (N, 230, 4)
    x_test = np.transpose(x_test_onehot, (0, 2, 1)).astype(np.float32)  # (N, 4, 230)
    print(f"Test set: {x_test.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    needs_oracle = any(t in ORACLE_TESTS for t in tests)
    y_test = None
    if needs_oracle:
        oracle = load_legnet_oracle(args.oracle, device, config_path=args.oracle_config)
        print("\nPredicting oracle on test sequences...")
        y_test = oracle.predict(x_test, batch_size=args.batch_size)  # (N, 1)

    results = {t: {} for t in tests}

    for npz_path in sample_files:
        name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"\n=== {name} ===")

        x_gen_onehot = np.load(npz_path)["arr_0"]  # (N, 230, 4)
        x_gen = np.transpose(x_gen_onehot, (0, 2, 1)).astype(np.float32)

        if needs_oracle:
            y_gen = oracle.predict(x_gen, batch_size=args.batch_size)

        if "fidelity_mse" in tests:
            mse = compute_fidelity_mse(y_test, y_gen)
            results["fidelity_mse"][name] = mse
            print(f"  fidelity_mse    = {mse:.6f}")

        if "ks_statistic" in tests:
            ks = compute_ks_statistic(y_test, y_gen)
            results["ks_statistic"][name] = ks
            print(f"  ks_statistic    = {ks:.6f}")

        if "js_k6" in tests:
            js = compute_js_spectrum(x_test, x_gen, ks=(6,))
            results["js_k6"][name] = js[6]
            print(f"  js_distance k=6 = {js[6]:.6f}")

        if "auroc" in tests:
            auroc = compute_auroc(x_test, x_gen, device=device)
            results["auroc"][name] = auroc
            print(f"  auroc           = {auroc:.6f}")

    names = sorted(next(iter(results.values())).keys())
    for metric, vals in results.items():
        csv_path = os.path.join(args.output_dir, f"{metric}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + names)
            w.writerow([metric] + [vals[n] for n in names])
        print(f"\nSaved {csv_path}")

    print("\n=== Summary ===")
    for metric, vals in results.items():
        arr = [vals[n] for n in names]
        print(f"  {metric}: mean={np.mean(arr):.6f}  ({', '.join(f'{v:.4f}' for v in arr)})")


if __name__ == "__main__":
    main()
