"""Evaluate generated K562 sequences against the real test split.

Owns everything K562-specific:
    * loading the H5 real-data layout (onehot_test -> (N, 4, 230))
    * loading the LegNet oracle
    * iterating over sample replicates (sample_*.npz or samples.npz) in --samples-dir

Delegates the metric math to d3_dna.D3Evaluator (no external pipeline).

Usage:
    python evaluate.py --samples-dir generated --data <H5> --oracle <ckpt>
    python evaluate.py --samples-dir generated --data <H5> --oracle <ckpt> --tests mse,ks
    python evaluate.py --samples-dir generated --data <H5> --oracle <ckpt> --kmer-ks 1-7
"""

import argparse
import csv
import glob
import json
import os

import h5py
import numpy as np
import torch

from d3_dna import D3Evaluator
from oracle import load as load_legnet_oracle


def parse_ks(spec: str):
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = [int(v) for v in spec.split("-")]
        return list(range(lo, hi + 1))
    return [int(v) for v in spec.split(",") if v]


def _load_k562_real(h5_path: str) -> np.ndarray:
    """Load the K562 test split one-hot (N, 4, 230) from the H5 file."""
    with h5py.File(h5_path, "r") as f:
        onehot = np.array(f["onehot_test"])  # (N, 230, 4)
    return np.transpose(onehot, (0, 2, 1)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Evaluate generated K562 sequences via D3Evaluator")
    p.add_argument("--samples-dir", default="generated",
                   help="Directory containing sample_*.npz (or samples.npz) replicates")
    p.add_argument("--data", required=True, help="Path to K562 H5 data file")
    p.add_argument("--oracle", required=True, help="Path to LegNet oracle checkpoint")
    p.add_argument("--oracle-config", default=None,
                   help="Optional LegNet config JSON path (defaults to oracle.DEFAULT_CONFIG)")
    p.add_argument("--output-dir", default="eval_results", help="Output directory")
    p.add_argument("--tests", default="mse,ks,js,auroc",
                   help="Comma-separated subset of {mse,ks,js,auroc}")
    p.add_argument("--kmer-ks", default="6",
                   help="k-mer length(s) for JS. Single k='6' reports that k; "
                        "interval '1-7' or list '3,6,7' reports the mean.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for oracle inference")
    args = p.parse_args()

    assert os.path.isdir(args.samples_dir), f"Samples directory not found: {args.samples_dir}"
    sample_files = sorted(glob.glob(os.path.join(args.samples_dir, "sample*.npz")))
    assert sample_files, f"No sample*.npz files found in {args.samples_dir}"
    print(f"Found {len(sample_files)} sample file(s) in {args.samples_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    ks = parse_ks(args.kmer_ks)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    real = _load_k562_real(args.data)
    print(f"[k562] real test split: shape={real.shape}")

    oracle = None
    if any(t in ("mse", "ks") for t in tests):
        oracle = load_legnet_oracle(args.oracle, device, config_path=args.oracle_config)

    ev = D3Evaluator(tests=tests, device=device)

    per_replicate: dict[str, dict] = {}
    for npz_path in sample_files:
        name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"\n=== {name} ===")
        out_json = os.path.join(args.output_dir, f"{name}.json")
        results = ev.evaluate(
            samples=npz_path,
            real_data=real,
            oracle=oracle,
            tests=tests,
            kmer_ks=ks,
            output_path=out_json,
        )
        per_replicate[name] = results

    # Aggregate: mean of each numeric metric across replicates.
    metric_keys = sorted({k for r in per_replicate.values() for k in r.keys()})
    aggregate = {
        k: float(np.mean([r[k] for r in per_replicate.values() if k in r]))
        for k in metric_keys
    }

    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["replicate", *metric_keys])
        for name in sorted(per_replicate):
            writer.writerow([name, *(per_replicate[name].get(k, "") for k in metric_keys)])
        writer.writerow(["mean", *(aggregate[k] for k in metric_keys)])

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"per_replicate": per_replicate, "mean": aggregate}, f, indent=2)

    print("\n=== Mean across replicates ===")
    for k in metric_keys:
        print(f"  {k}: {aggregate[k]:.6f}")
    print(f"\nSaved: {csv_path}, {summary_path}, and per-replicate JSONs in {args.output_dir}/")


if __name__ == "__main__":
    main()
