"""Evaluate generated promoter sequences.

Owns everything promoter-specific:
    * loading the NPZ real-data layout (N, 1024, 6) -> one-hot channels 0..3
    * loading the SEI oracle (with H3K4me3 masking + strand averaging)
    * optional 5-per-TSS tiling of the real set (DDSM protocol)

Delegates only the metric math to d3_dna.

Usage:
    python evaluate.py --samples generated/samples.npz
    python evaluate.py --samples samples.npz --tests mse,ks --kmer-ks 1-7
    python evaluate.py --samples samples.npz --paired-repeat 5    # DDSM 5-per-TSS
"""

import argparse
import json
import os

import numpy as np
import torch

from d3_dna import D3Evaluator
from oracle import load as load_sei_oracle


DEFAULT_REAL_DATA = "/grid/koo/home/shared/d3/data/promoter/Promoter_data.npz"


def _load_promoter_real(path: str) -> np.ndarray:
    """Load the promoter test split one-hot (N, 4, L) from the 40k NPZ."""
    return np.load(path)["test"][:, :, :4].transpose(0, 2, 1).astype(np.float32)


def parse_ks(spec: str):
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = [int(v) for v in spec.split("-")]
        return list(range(lo, hi + 1))
    return [int(v) for v in spec.split(",") if v]


def main():
    p = argparse.ArgumentParser(description="Evaluate generated promoter sequences")
    p.add_argument("--samples", required=True, help="Path to generated samples NPZ")
    p.add_argument("--oracle",
                   default="/grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar",
                   help="Path to SEI oracle checkpoint")
    p.add_argument("--data", default=DEFAULT_REAL_DATA,
                   help="Real-data NPZ path (40k FANTOM test split)")
    p.add_argument("--output-dir", default="eval_results", help="Output directory")
    p.add_argument("--tests", default="mse,ks,js,auroc",
                   help="Comma-separated subset of {mse,ks,js,auroc}")
    p.add_argument("--kmer-ks", default="1-7",
                   help="k-mer lengths for JS divergence mean (e.g. '1-7' or '3,6,7')")
    p.add_argument("--paired-repeat", type=int, default=1,
                   help="Repeat each real seq N times before pairing (5 for DDSM 5-per-TSS protocol)")
    args = p.parse_args()

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    ks = parse_ks(args.kmer_ks)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    real = _load_promoter_real(args.data)
    if args.paired_repeat > 1:
        real = np.repeat(real, args.paired_repeat, axis=0)
        print(f"[promoter] real tiled x{args.paired_repeat} -> {real.shape}")

    oracle = None
    if any(t in ("mse", "ks") for t in tests):
        oracle = load_sei_oracle(args.oracle, device)

    ev = D3Evaluator(tests=tests, device=device)
    results = ev.evaluate(
        samples=args.samples,
        real_data=real,
        oracle=oracle,
        tests=tests,
        kmer_ks=ks,
        output_path=os.path.join(args.output_dir, "eval_results.json"),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
