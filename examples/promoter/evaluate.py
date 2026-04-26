"""Evaluate generated promoter sequences.

Owns everything promoter-specific:
    * loading the NPZ real-data layout (N, 1024, 6) -> one-hot channels 0..3
    * loading the SEI oracle (with H3K4me3 masking + strand averaging)
    * optional 5-per-TSS tiling of the real set (DDSM protocol)
    * iterating over sample replicates (samples.npz or sample_*.npz) in --samples-dir

Delegates only the metric math to d3_dna.D3Evaluator.

Data + oracle weights resolve through data.py: CLI override > local cache >
download from Zenodo.

Usage:
    python evaluate.py --samples-dir generated
    python evaluate.py --samples-dir generated --tests mse,ks --kmer-ks 1-7
    python evaluate.py --samples-dir generated --paired-repeat 5    # DDSM 5-per-TSS

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

import numpy as np
import torch
from omegaconf import OmegaConf

from d3_dna import D3Evaluator
from data import get_data_file, get_oracle_file, get_sei_features_file
from oracle import load as load_sei_oracle


def _load_promoter_real(path: str) -> np.ndarray:
    """Load the promoter test split one-hot (N, 4, L) from the 40k NPZ."""
    return np.load(path)["test"][:, :, :4].transpose(0, 2, 1).astype(np.float32)


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
    sei_features: Optional[str] = None,
    output_dir: str = "eval_results",
    tests: str = "mse,ks,js,auroc",
    kmer_ks: str = "6",
    paired_repeat: int = 1,
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
    real = _load_promoter_real(str(data_path))
    if paired_repeat > 1:
        real = np.repeat(real, paired_repeat, axis=0)
        print(f"[promoter] real tiled x{paired_repeat} -> {real.shape}")

    oracle = None
    if any(t in ("mse", "ks") for t in test_list):
        oracle_path = get_oracle_file(cfg, override=oracle_file)
        sei_features_path = get_sei_features_file(cfg, override=sei_features)
        oracle = load_sei_oracle(
            str(oracle_path), device, target_names_path=str(sei_features_path)
        )

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
    p = argparse.ArgumentParser(description="Evaluate generated promoter sequences")
    p.add_argument("--samples-dir", default="generated",
                   help="Directory containing sample*.npz replicates")
    p.add_argument("--config", default="config_transformer.yaml")
    p.add_argument("--data-file", default=None,
                   help="Override config; if absent, downloads from Zenodo.")
    p.add_argument("--oracle-file", default=None,
                   help="Override config; if absent, downloads from Zenodo.")
    p.add_argument("--sei-features", default=None,
                   help="Override config; defaults to vendored target.sei.names.")
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--tests", default="mse,ks,js,auroc",
                   help="Comma-separated subset of {mse,ks,js,auroc}")
    p.add_argument("--kmer-ks", default="6",
                   help="k-mer length(s) for JS divergence. Single k='6' reports that k "
                        "alone; interval '1-7' or list '3,6,7' reports the mean across them.")
    p.add_argument("--paired-repeat", type=int, default=1,
                   help="Repeat each real seq N times before pairing (5 for DDSM 5-per-TSS protocol)")
    args = p.parse_args()
    main(
        samples_dir=args.samples_dir,
        config=args.config,
        data_file=args.data_file,
        oracle_file=args.oracle_file,
        sei_features=args.sei_features,
        output_dir=args.output_dir,
        tests=args.tests,
        kmer_ks=args.kmer_ks,
        paired_repeat=args.paired_repeat,
    )
