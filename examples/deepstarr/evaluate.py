"""Evaluate generated DeepSTARR sequences against the real test split.

Owns everything DeepSTARR-specific:
    * loading the H5 real-data layout (onehot_test -> (N, 4, 249))
    * loading the DeepSTARR CNN oracle (dual-head: dev, hk)
    * iterating over sample replicates (sample_*.npz or samples.npz) in --samples-dir

Delegates the metric math to d3_dna.D3Evaluator. The MSE/KS metrics naturally
compose over the 2-D oracle output (mean reduces over the 2 heads).

Data + oracle weights resolve through data.py: CLI override > local cache >
download from Zenodo.

Usage:
    python evaluate.py --samples-dir generated
    python evaluate.py --samples-dir generated --tests mse,ks --kmer-ks 1-7
    python evaluate.py --samples-dir generated --config config_conv.yaml

Importable:
    from evaluate import main
    main(samples_dir="generated", tests="mse,ks")
"""

import argparse
import csv
import glob
import json
import os
from typing import Optional

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from d3_dna import D3Evaluator
from data import get_data_file, get_oracle_file
from oracle import load as load_deepstarr_oracle


def _load_deepstarr_real(h5_path: str) -> np.ndarray:
    """Load the DeepSTARR test split one-hot (N, 4, 249) from the H5 file.

    The Zenodo H5 already stores X_* in NCHW; no transpose needed.
    """
    with h5py.File(h5_path, "r") as f:
        x = np.array(f["X_test"])  # (N, 4, 249) float32
    return x.astype(np.float32)


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
    real = _load_deepstarr_real(str(data_path))
    print(f"[deepstarr] real test split: shape={real.shape}")

    oracle = None
    if any(t in ("mse", "ks") for t in test_list):
        oracle_path = get_oracle_file(cfg, override=oracle_file)
        oracle = load_deepstarr_oracle(str(oracle_path), device)

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
    p = argparse.ArgumentParser(description="Evaluate generated DeepSTARR sequences via D3Evaluator")
    p.add_argument("--samples-dir", default="generated",
                   help="Directory containing sample*.npz replicates")
    p.add_argument("--config", default="config_transformer.yaml")
    p.add_argument("--data-file", default=None,
                   help="Override config; if absent, downloads from Zenodo.")
    p.add_argument("--oracle-file", default=None,
                   help="Override config; if absent, downloads from Zenodo.")
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--tests", default="mse,ks,js,auroc",
                   help="Comma-separated subset of {mse,ks,js,auroc}")
    p.add_argument("--kmer-ks", default="6",
                   help="k-mer length(s) for JS divergence. Single k='6' reports that k "
                        "alone; interval '1-7' or list '3,6,7' reports the mean across them.")
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
