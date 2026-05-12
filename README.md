# D3-DNA: DNA Discrete Diffusion

A PyPI package for training and sampling discrete diffusion models on DNA sequences.

This package provides a clean, reusable implementation of the D3 (DNA Discrete Diffusion) framework from [D3-DNA-Discrete-Diffusion](https://github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion). The original repository contains the full research codebase, experiment scripts, and analysis pipelines. This package extracts the core model, training, and sampling components into a pip-installable library suitable for integration into new projects.

## Installation

```bash
# Core package
pip install d3-dna

# With flash attention (faster training on long sequences)
pip install d3-dna[flash]

# With Weights & Biases logging
pip install d3-dna[logging]

# Everything
pip install d3-dna[all]
```

**GPU acceleration**: `d3-dna[flash]` installs [flash attention](https://github.com/Dao-AILab/flash-attention) for faster, more memory-efficient training on long sequences. Without it, the package uses PyTorch's built-in scaled dot-product attention (SDPA) — same model quality, just slower for long inputs.

## Quickstart

### 1. Define your dataset

```python
import torch
from torch.utils.data import Dataset

class MyDNADataset(Dataset):
    def __init__(self, h5_path, split='train'):
        import h5py
        with h5py.File(h5_path, 'r') as f:
            self.X = torch.tensor(f[f'X_{split}'][:]).argmax(dim=1)  # one-hot to indices
            self.y = torch.tensor(f[f'Y_{split}'][:])

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]
```

### 2. Write a config

```yaml
dataset:
  name: my_dataset
  sequence_length: 249
  num_classes: 4
  signal_dim: 2

ngpus: 1
tokens: 4

model:
  architecture: transformer
  hidden_size: 256
  cond_dim: 128
  n_blocks: 8
  n_heads: 8
  dropout: 0.1
  class_dropout_prob: 0.1

training:
  batch_size: 128
  accum: 1
  max_epochs: 300
  ema: 0.9999

# ... see examples/minimal/config.yaml for full template
```

### 3. Train

```python
from d3_dna import D3Trainer

trainer = D3Trainer('config.yaml')
trainer.fit(
    train_dataset=MyDNADataset('data.h5', 'train'),
    val_dataset=MyDNADataset('data.h5', 'valid'),
)
```

### 4. Sample

```python
from d3_dna import D3Sampler
from d3_dna.models import TransformerModel
from omegaconf import OmegaConf

cfg = OmegaConf.load('config.yaml')
model = TransformerModel(cfg)
sampler = D3Sampler(cfg)

sequences = sampler.generate(
    checkpoint='experiments/checkpoints/last.ckpt',
    model=model,
    num_samples=1000,
)
sampler.save(sequences, 'generated.fasta')
```

## API Reference

| Class | Purpose |
|---|---|
| `D3Trainer` | Train a D3 model: `trainer.fit(train_ds, val_ds)` |
| `D3Sampler` | Generate sequences: `sampler.generate(ckpt, model, n)` |
| `D3Evaluator` | Evaluate with oracle: subclass and implement `load_oracle_model()` |
| `TransformerModel` | D3-Tran architecture (config-driven) |
| `ConvolutionalModel` | D3-Conv architecture (config-driven) |

## Mixed precision

Both training and sampling default to a per-architecture autocast policy, picked by `d3_dna.modules.precision.precision_for_cfg`:

| Architecture | Lightning precision | Autocast dtype | GradScaler |
|---|---|---|---|
| `transformer` | `bf16-mixed` | `torch.bfloat16` | not used (bf16 has fp32 range) |
| `convolutional` | `16-mixed` | `torch.float16` | installed automatically by Lightning |

The same dtype flows through `get_score_fn` so the loss path and the sampling path share a single autocast policy. `LayerNorm` opts out of autocast and runs in fp32; `score.exp()` in the predictor path is on PyTorch's fp32-cast list, so the score and all post-model sampler arithmetic land in fp32 regardless of architecture.

To override (e.g. when a checkpoint was trained under a different policy), set `cfg.training.precision: '16-mixed'` or `'bf16-mixed'` on the config. **Promoter is a known exception** — `examples/promoter/config_transformer.yaml` overrides the default back to `16-mixed` because the public `D3_Tran_Promoter.ckpt` on Zenodo was trained / validated under fp16 and degrades sharply if sampled in bf16. New transformer training runs in other examples should keep the bf16-mixed default.

## Architecture

```
d3_dna/
├── models/          # TransformerModel, ConvolutionalModel, EMA
├── diffusion.py     # Noise schedules, transition graphs, losses
├── sampling.py      # PC sampler, predictors, D3Sampler
├── trainer.py       # Lightning module, D3Trainer
├── evaluator.py     # SP-MSE callback, D3Evaluator
└── io.py            # Checkpoint loading, data utilities
```
