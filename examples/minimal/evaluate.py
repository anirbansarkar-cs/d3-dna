"""
Template — evaluate.py for a new D3 example.

Reference: examples/k562/evaluate.py.

The only dataset-specific bit is `_load_real(...)` — how to read your test-split
one-hot tensor in shape (N, 4, L). Everything else is shared boilerplate.
"""

import argparse
import csv
import glob
import json
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from d3_dna import D3Evaluator
from data import get_data_file, get_oracle_file
from oracle import load as load_minimal_oracle


def _load_real(data_path: str) -> np.ndarray:
    """Load the test split as one-hot (N, 4, sequence_length) float32.

    Examples to copy from:
        examples/k562/evaluate.py:_load_k562_real     — H5, transpose NHWC→NCHW
        examples/promoter/evaluate.py:_load_promoter_real — NPZ
        examples/deepstarr/evaluate.py:_load_deepstarr_real — H5, already NCHW
    """
    raise NotImplementedError("Read your test split, return (N, 4, L) np.float32.")


def parse_ks(spec: str):
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = [int(v) for v in spec.split("-")]
        return list(range(lo, hi + 1))
    return [int(v) for v in spec.split(",") if v]


def main(
    samples_dir: str = "generated",
    *,
    config: str = "config_transformer.yaml",
    data_file: Optional[str] = None,
    oracle_file: Optional[str] = None,
    output_dir: str = "eval_results",
    tests: str = "mse,ks,js,auroc",
    kmer_ks: str = "6",
) -> None:
    cfg = OmegaConf.load(config)

    assert os.path.isdir(samples_dir), f"Samples directory not found: {samples_dir}"
    sample_files = sorted(glob.glob(os.path.join(samples_dir, "sample*.npz")))
    assert sample_files, f"No sample*.npz files found in {samples_dir}"
    print(f"Found {len(sample_files)} sample file(s) in {samples_dir}")

    os.makedirs(output_dir, exist_ok=True)

    test_list = [t.strip() for t in tests.split(",") if t.strip()]
    ks = parse_ks(kmer_ks)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = get_data_file(cfg, override=data_file)
    real = _load_real(str(data_path))
    print(f"[minimal] real test split: shape={real.shape}")

    oracle = None
    if any(t in ("mse", "ks") for t in test_list):
        oracle_path = get_oracle_file(cfg, override=oracle_file)
        oracle = load_minimal_oracle(str(oracle_path), device)

    ev = D3Evaluator(tests=test_list, device=device)

    per_replicate: dict[str, dict] = {}
    for npz_path in sample_files:
        name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"\n=== {name} ===")
        out_json = os.path.join(output_dir, f"{name}.json")
        results = ev.evaluate(
            samples=npz_path,
            real_data=real,
            oracle=oracle,
            tests=test_list,
            kmer_ks=ks,
            output_path=out_json,
        )
        per_replicate[name] = results

    metric_keys = sorted({k for r in per_replicate.values() for k in r.keys()})
    aggregate = {
        k: float(np.mean([r[k] for r in per_replicate.values() if k in r]))
        for k in metric_keys
    }

    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["replicate", *metric_keys])
        for name in sorted(per_replicate):
            writer.writerow([name, *(per_replicate[name].get(k, "") for k in metric_keys)])
        writer.writerow(["mean", *(aggregate[k] for k in metric_keys)])

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"per_replicate": per_replicate, "mean": aggregate}, f, indent=2)

    print("\n=== Mean across replicates ===")
    for k in metric_keys:
        print(f"  {k}: {aggregate[k]:.6f}")
    print(f"\nSaved: {csv_path}, {summary_path}, and per-replicate JSONs in {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate generated sequences via D3Evaluator")
    p.add_argument("--samples-dir", default="generated")
    p.add_argument("--config", default="config_transformer.yaml")
    p.add_argument("--data-file", default=None)
    p.add_argument("--oracle-file", default=None)
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--tests", default="mse,ks,js,auroc")
    p.add_argument("--kmer-ks", default="6")
    args = p.parse_args()
    main(
        samples_dir=args.samples_dir,
        config=args.config,
        data_file=args.data_file,
        oracle_file=args.oracle_file,
        output_dir=args.output_dir,
        tests=args.tests,
        kmer_ks=args.kmer_ks,
    )
