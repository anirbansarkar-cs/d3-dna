"""
Generate sequences from a trained Promoter checkpoint.

For loading pretrained hybrid SEDD checkpoints produced outside of d3_dna's
native training pipeline, use examples/promoter/legacy/sample_hybrid.py.

Usage:
    python sample.py --checkpoint /path/to/model.ckpt
    python sample.py --checkpoint /path/to/model.ckpt --num-samples 100 --steps 128
    python sample.py --checkpoint /path/to/model.ckpt --use-test-labels

Importable:
    from sample import main
    main("/path/to/model.ckpt", num_samples=64, steps=128)
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
from data import PromoterDataset, get_data_file, get_checkpoint_file


def main(
    checkpoint: Optional[str] = None,
    *,
    config: str = "config_transformer.yaml",
    output_dir: str = "generated",
    use_test_labels: bool = False,
    num_samples: Optional[int] = None,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    predictor: Optional[str] = None,
    paired_repeat: Optional[int] = None,
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
        "paired_repeat": paired_repeat,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg.sampling[k] = v

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_test_labels:
        data_path = get_data_file(cfg, override=data_file)
        test_ds = PromoterDataset(data_path, split="test")
        labels = test_ds.y  # (N, 1024, 1)
        repeat = cfg.sampling.get("paired_repeat", 1)
        if repeat > 1:
            labels = labels.repeat_interleave(repeat, dim=0)
        n = len(labels)
        print(f"Using test set labels: {n} samples (paired_repeat={repeat}), shape {labels.shape}")
    else:
        n = cfg.sampling.num_samples
        seq_len = cfg.dataset.sequence_length
        labels = torch.randn(n, seq_len, 1) * 2.0
        print(f"Using random labels: {n} samples, shape {labels.shape}")

    arch = getattr(cfg.model, "architecture", "transformer")
    if arch == "convolutional":
        model = ConvolutionalModel(cfg)
    else:
        model = TransformerModel(cfg)

    sampler = D3Sampler(cfg)
    sampler.load(checkpoint=str(checkpoint_path), model=model, device=device)

    os.makedirs(output_dir, exist_ok=True)

    seqs = sampler.generate_batched(
        num_samples=n,
        labels=labels,
        batch_size=cfg.sampling.batch_size,
        steps=cfg.sampling.steps,
    )

    onehot = F.one_hot(seqs.long(), num_classes=4).float().numpy()  # (N, 1024, 4)
    npz_path = os.path.join(output_dir, "samples.npz")
    np.savez(npz_path, arr_0=onehot)

    fasta_path = os.path.join(output_dir, "samples.fasta")
    sampler.save(seqs, fasta_path, format="fasta")

    print(f"Saved {len(seqs)} sequences:")
    print(f"  NPZ:   {npz_path} shape={onehot.shape}")
    print(f"  FASTA: {fasta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Override config; if absent, downloads from Zenodo.")
    p.add_argument("--config", default="config_transformer.yaml", help="Config YAML path")
    p.add_argument("--output-dir", default="generated", help="Output directory")
    p.add_argument("--use-test-labels", action="store_true",
                   help="Use test-set labels for conditional generation")
    p.add_argument("--num-samples", type=int, default=None, help="Override cfg.sampling.num_samples")
    p.add_argument("--steps", type=int, default=None, help="Override cfg.sampling.steps")
    p.add_argument("--batch-size", type=int, default=None, help="Override cfg.sampling.batch_size")
    p.add_argument("--predictor", default=None, help="Override cfg.sampling.predictor")
    p.add_argument("--paired-repeat", type=int, default=None,
                   help="Override cfg.sampling.paired_repeat (5 for DDSM 5-per-TSS)")
    p.add_argument("--data-file", default=None,
                   help="Only used with --use-test-labels; override config / Zenodo download.")
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
        paired_repeat=args.paired_repeat,
        data_file=args.data_file,
    )
