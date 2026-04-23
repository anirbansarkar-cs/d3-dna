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
from d3_dna.models import TransformerModel, ConvolutionalModel
from data import PromoterDataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
parser.add_argument("--steps", type=int, default=128, help="Number of sampling steps")
parser.add_argument("--num-samples", type=int, default=10, help="Number of sequences to generate (ignored with --use-test-labels)")
parser.add_argument("--batch-size", type=int, default=64, help="Sequences per GPU batch")
parser.add_argument("--output-dir", type=str, default="generated", help="Output directory")
parser.add_argument("--use-test-labels", action="store_true", help="Use test set labels for conditional generation")
parser.add_argument("--paired-repeat", type=int, default=1, help="Generate N samples per TSS (5 for DDSM 5-per-TSS protocol)")
parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML path")
parser.add_argument("--predictor", type=str, default=None, help="Override sampling predictor (euler or analytic)")
parser.add_argument("--hybrid-shim", action="store_true",
                    help="Use HybridSEDD shim for loading the colleague's dual-tower tran checkpoint")
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
if args.predictor:
    cfg.sampling.predictor = args.predictor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Determine labels
if args.use_test_labels:
    test_ds = PromoterDataset(cfg.paths.data_file, split="test")
    labels = test_ds.y  # (N, 1024, 1)
    if args.paired_repeat > 1:
        labels = labels.repeat_interleave(args.paired_repeat, dim=0)
    num_samples = len(labels)
    print(f"Using test set labels: {num_samples} samples (paired_repeat={args.paired_repeat}), shape {labels.shape}")
else:
    num_samples = args.num_samples
    seq_len = cfg.dataset.sequence_length
    labels = torch.randn(num_samples, seq_len, 1) * 2.0
    print(f"Using random labels: {num_samples} samples, shape {labels.shape}")

if args.hybrid_shim:
    from hybrid_shim import HybridSEDD
    model = HybridSEDD(cfg)
    # Validate the shim matches the checkpoint BEFORE sampler.load (which uses strict=False).
    _raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _ckpt_keys = set(_raw_ckpt["model"].keys())
    _model_keys = set(model.state_dict().keys())
    _missing = _model_keys - _ckpt_keys
    _unexpected = _ckpt_keys - _model_keys
    assert not _missing, f"HybridSEDD expects keys absent from checkpoint: {sorted(_missing)[:10]}"
    assert not _unexpected, f"Checkpoint has keys absent from HybridSEDD: {sorted(_unexpected)[:10]}"
    _shadow = _raw_ckpt["ema"]["shadow_params"]
    _trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(_shadow) == _trainable, \
        f"EMA shadow_params count {len(_shadow)} != trainable params {_trainable}; registration order wrong"
    print(f"[hybrid-shim] checkpoint load validated: {len(_ckpt_keys)} keys, {_trainable} trainable params")
    del _raw_ckpt, _shadow
else:
    arch = getattr(cfg.model, 'architecture', 'transformer')
    if arch == 'convolutional':
        model = ConvolutionalModel(cfg)
    else:
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
