"""
Generate sequences from a trained K562 checkpoint.

Produces 5 replicates of test-conditioned samples in both NPZ (one-hot) and FASTA formats.

Usage:
    python sample.py
    python sample.py --checkpoint outputs/k562/checkpoints/model-epoch=263.ckpt --steps 230
    python sample.py --replicates 1  # Quick single-replicate test
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from d3_dna import D3Sampler
from d3_dna.models import TransformerModel
from data import K562Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
parser.add_argument("--steps", type=int, default=20, help="Number of sampling steps")
parser.add_argument("--replicates", type=int, default=5, help="Number of sample replicates")
parser.add_argument("--output-dir", type=str, default="generated", help="Output directory")
args = parser.parse_args()

cfg = OmegaConf.load("config.yaml")

# Find checkpoint
if args.checkpoint:
    ckpt = args.checkpoint
else:
    ckpts = glob.glob("outputs/k562/checkpoints/model-*.ckpt")
    assert ckpts, "No checkpoints found. Run train.py first."
    # Pick checkpoint with lowest val_loss
    ckpt = min(ckpts, key=lambda p: float(os.path.basename(p).split("val_loss=")[1].replace(".ckpt", "")))
print(f"Using checkpoint: {ckpt}")

# Load test labels for conditional generation
test_ds = K562Dataset(cfg.paths.data_file, split="test")
labels = test_ds.y  # (N_test, 1)
print(f"Generating {args.replicates} replicates of {len(labels)} sequences ({args.steps} steps)")

model = TransformerModel(cfg)
sampler = D3Sampler(cfg)
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(args.output_dir, exist_ok=True)

for rep in range(args.replicates):
    print(f"\n--- Replicate {rep} ---")
    seqs = sampler.generate(
        checkpoint=ckpt,
        model=model,
        num_samples=len(labels),
        labels=labels,
        steps=args.steps,
        device=device,
    )

    # Save as one-hot NPZ (evaluation pipeline format)
    onehot = F.one_hot(seqs.long(), num_classes=4).float().numpy()  # (N, 230, 4)
    npz_path = os.path.join(args.output_dir, f"sample_{rep}.npz")
    np.savez(npz_path, arr_0=onehot)

    # Save as FASTA
    fasta_path = os.path.join(args.output_dir, f"sample_{rep}.fasta")
    sampler.save(seqs, fasta_path, format="fasta")

    print(f"Saved {npz_path} shape={onehot.shape}, {fasta_path}")

print(f"\nDone. All replicates saved to {args.output_dir}/")
