"""
Generate sequences from a trained Promoter checkpoint.

Usage:
    python sample.py --checkpoint /path/to/model.ckpt
    python sample.py --checkpoint /path/to/model.ckpt --num-samples 100 --steps 128
    python sample.py --checkpoint /path/to/model.ckpt --use-test-labels
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from d3_dna import D3Sampler
from d3_dna.models import TransformerModel
from data import PromoterDataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
parser.add_argument("--steps", type=int, default=128, help="Number of sampling steps")
parser.add_argument("--num-samples", type=int, default=10, help="Number of sequences to generate (ignored with --use-test-labels)")
parser.add_argument("--batch-size", type=int, default=64, help="Sequences per GPU batch")
parser.add_argument("--output-dir", type=str, default="generated", help="Output directory")
parser.add_argument("--use-test-labels", action="store_true", help="Use test set labels for conditional generation")
parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML path")
args = parser.parse_args()

cfg = OmegaConf.load(args.config)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Determine labels
if args.use_test_labels:
    test_ds = PromoterDataset(cfg.paths.data_file, split="test")
    labels = test_ds.y  # (N, 1024, 1)
    num_samples = len(test_ds)
    print(f"Using test set labels: {num_samples} samples, shape {labels.shape}")
else:
    num_samples = args.num_samples
    seq_len = cfg.dataset.sequence_length
    labels = torch.randn(num_samples, seq_len, 1) * 2.0
    print(f"Using random labels: {num_samples} samples, shape {labels.shape}")

model = TransformerModel(cfg)
sampler = D3Sampler(cfg)
sampler.load(checkpoint=args.checkpoint, model=model, device=device)

os.makedirs(args.output_dir, exist_ok=True)

seqs = sampler.generate_batched(
    num_samples=num_samples,
    labels=labels,
    batch_size=args.batch_size,
    steps=args.steps,
)

# Save as one-hot NPZ
onehot = F.one_hot(seqs.long(), num_classes=4).float().numpy()  # (N, 1024, 4)
npz_path = os.path.join(args.output_dir, "samples.npz")
np.savez(npz_path, arr_0=onehot)

# Save as FASTA
fasta_path = os.path.join(args.output_dir, "samples.fasta")
sampler.save(seqs, fasta_path, format="fasta")

print(f"Saved {len(seqs)} sequences:")
print(f"  NPZ:   {npz_path} shape={onehot.shape}")
print(f"  FASTA: {fasta_path}")
