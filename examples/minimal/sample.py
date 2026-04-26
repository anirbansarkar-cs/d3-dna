"""
Template — sample.py for a new D3 example.

Reference: examples/k562/sample.py. Adjust the label-loading branch (test
labels vs random) for your dataset's signal_dim and conditioning style.
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from d3_dna import D3Sampler
from d3_dna.models import TransformerModel, ConvolutionalModel
from data import MinimalDataset, get_data_file, get_checkpoint_file


def main(
    checkpoint: Optional[str] = None,
    *,
    config: str = "config_transformer.yaml",
    output_dir: str = "generated",
    use_test_labels: bool = True,
    num_samples: Optional[int] = None,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    predictor: Optional[str] = None,
    replicates: int = 5,
    rep_offset: int = 0,
    data_file: Optional[str] = None,
) -> None:
    cfg = OmegaConf.load(config)
    checkpoint_path = get_checkpoint_file(cfg, override=checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")

    overrides = {
        "num_samples": num_samples,
        "steps": steps,
        "batch_size": batch_size,
        "predictor": predictor,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg.sampling[k] = v

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_test_labels:
        data_path = get_data_file(cfg, override=data_file)
        test_ds = MinimalDataset(data_path, split="test")
        labels = test_ds.y
        n = len(labels)
        print(f"Using test set labels: {n} samples, shape {labels.shape}")
    else:
        n = cfg.sampling.num_samples
        labels = torch.randn(n, cfg.dataset.signal_dim) * 2.0
        print(f"Using random labels: {n} samples, shape {labels.shape}")

    arch = getattr(cfg.model, "architecture", "transformer")
    if arch == "convolutional":
        model = ConvolutionalModel(cfg)
    else:
        model = TransformerModel(cfg)

    sampler = D3Sampler(cfg)
    sampler.load(checkpoint=str(checkpoint_path), model=model, device=device)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {replicates} replicate(s) of {n} sequences ({cfg.sampling.steps} steps)")
    for rep_idx in range(replicates):
        rep = rep_idx + rep_offset
        print(f"\n--- Replicate {rep} ---")

        seqs = sampler.generate_batched(
            num_samples=n,
            labels=labels,
            batch_size=cfg.sampling.batch_size,
            steps=cfg.sampling.steps,
        )

        onehot = F.one_hot(seqs.long(), num_classes=4).float().numpy()  # (N, L, 4)
        npz_path = os.path.join(output_dir, f"sample_{rep}.npz")
        np.savez(npz_path, arr_0=onehot)

        fasta_path = os.path.join(output_dir, f"sample_{rep}.fasta")
        sampler.save(seqs, fasta_path, format="fasta")

        print(f"  Saved {npz_path} shape={onehot.shape}, {fasta_path}")

    print(f"\nDone. All replicates saved to {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Override config; if absent, downloads from Zenodo.")
    p.add_argument("--config", default="config_transformer.yaml")
    p.add_argument("--output-dir", default="generated")
    label_group = p.add_mutually_exclusive_group()
    label_group.add_argument("--use-test-labels", dest="use_test_labels",
                             action="store_true", default=True,
                             help="Condition on test-set labels (default)")
    label_group.add_argument("--random-labels", dest="use_test_labels",
                             action="store_false",
                             help="Condition on random scalar labels instead")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--predictor", default=None)
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--rep-offset", type=int, default=0)
    p.add_argument("--data-file", default=None)
    args = p.parse_args()
    main(
        checkpoint=args.checkpoint,
        config=args.config,
        output_dir=args.output_dir,
        use_test_labels=args.use_test_labels,
        num_samples=args.num_samples,
        steps=args.steps,
        batch_size=args.batch_size,
        predictor=args.predictor,
        replicates=args.replicates,
        rep_offset=args.rep_offset,
        data_file=args.data_file,
    )
