# d3-dna Sampling Guide

Tested on: RHEL 8, SLURM cluster with H100 GPUs, conda env `d3-dna`, 2026-04-09.

## Prerequisites

- Installed `d3-dna` environment (see `INSTALLATION.md`)
- Access to a GPU node (H100 required for flash attention)
- A trained checkpoint and matching config

## Quick start

```bash
srun --partition=gpuq --gres=gpu:1 --constraint="h100" --time=00:10:00 bash -c '
  conda activate d3-dna
  cd /path/to/d3-dna/examples/promoter
  python sample.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/promoter_09242025/model-epoch=175-val_loss=1119.9065.ckpt \
    --num-samples 10 \
    --steps 128
'
```

## Global vs per-position conditioning

D3-DNA supports two conditioning modes. The mode is determined by the **shape of the label tensor** passed to the model, not by a config flag.

### Global conditioning (K562, LentiMPRA)

Labels are a single value (or vector) per sequence, broadcast to all positions.

- **Label shape**: `(batch, signal_dim)` — e.g., `(64, 1)` for K562
- **Config**: `dataset.signal_dim: 1` (or 2, 3, etc.)
- **Use case**: One regulatory activity measurement per sequence
- **Data format**: Labels stored separately from sequences (e.g., in H5 as `Y_train`)

```python
# Global: one scalar per sequence
labels = torch.randn(num_samples, 1)  # (B, signal_dim)
```

### Per-position conditioning (Promoter)

Labels are a value at every base pair position, added element-wise to token embeddings.

- **Label shape**: `(batch, seq_len, signal_dim)` — e.g., `(64, 1024, 1)` for Promoter
- **Config**: `dataset.signal_dim: 1`
- **Use case**: Per-position regulatory activity (e.g., transcription initiation likelihood per base)
- **Data format**: Labels stored as extra channels alongside one-hot DNA in NPZ

```python
# Per-position: one value at each of 1024 positions
labels = torch.randn(num_samples, 1024, 1)  # (B, seq_len, signal_dim)
```

### How it works in the model

The `EmbeddingLayer` in `d3_dna/models/transformer.py` handles both cases automatically:

```python
def forward(self, x, y):
    vocab_embed = self.embedding[x]                          # (B, seq_len, hidden)
    signal_embed = self.signal_embedding(y.to(torch.float32))  # Linear(signal_dim, hidden)
    if signal_embed.dim() == 2:
        signal_embed = signal_embed.unsqueeze(1)             # (B, 1, hidden) for broadcast
    return torch.add(vocab_embed, signal_embed)
```

- **2D signal** `(B, hidden)` → unsqueezed to `(B, 1, hidden)` → broadcast-added (global)
- **3D signal** `(B, seq_len, hidden)` → added directly (per-position)

The `signal_dim` config value is the same in both cases — it controls the input dimension of the linear projection, not whether conditioning is global or per-position. That distinction comes entirely from the label tensor shape.

### Config differences

Both modes use the same config structure. The only meaningful difference for sampling is how you prepare labels:

| Setting | K562 (global) | Promoter (per-position) |
|---|---|---|
| `dataset.signal_dim` | 1 | 1 |
| `dataset.sequence_length` | 230 | 1024 |
| Label shape at inference | `(B, 1)` | `(B, 1024, 1)` |
| Data source for labels | Separate Y array | Channel 4 of NPZ (per-position) |

## Promoter data format

The promoter NPZ file (`Promoter_data.npz`) has shape `(N, 1024, 6)`:

| Channels | Content |
|---|---|
| 0-3 | One-hot DNA (A, C, G, T) |
| 4 | Per-position regulatory activity (used as label) |
| 5 | Second regulatory signal (unused by this model) |

## Creating a config from a checkpoint

Checkpoints embed the full training config in `hyper_parameters`. To extract it:

```python
import torch
from omegaconf import OmegaConf

ckpt = torch.load('model.ckpt', map_location='cpu', weights_only=False)
cfg = ckpt['hyper_parameters']['cfg']
print(OmegaConf.to_yaml(OmegaConf.create(dict(cfg))))
```

Save that output as your `config.yaml`. Override `ngpus: 1` for single-GPU sampling.

## Issues encountered and fixes

### 1. EmbeddingLayer only supported global conditioning

**Problem**: The d3-dna package's `EmbeddingLayer.forward()` always applied `signal_embed[:, None, :]`, which assumes global labels `(B, signal_dim)`. Promoter uses per-position labels `(B, 1024, 1)`, which would produce a wrong 4D tensor.

**Fix**: Changed `EmbeddingLayer.forward()` to check `signal_embed.dim()` and only unsqueeze for 2D (global) inputs. See `d3_dna/models/transformer.py`.

### 2. Package not on PyPI

**Problem**: `pip install d3-dna` fails — package not published yet.

**Fix**: Install from local source: `pip install -e /path/to/d3-dna`

### 3. PyTorch version too new for cluster

**Problem**: Latest PyTorch (2.11.0, CUDA 13.0) drops V100 support and may not match cluster drivers.

**Fix**: Pin `torch>=2.0.0,<2.6.0` to get PyTorch 2.5.1 with CUDA 12.4.

### 4. Flash attention requires Ampere+ GPUs

**Problem**: `flash_attn_varlen_qkvpacked_func` raises `RuntimeError: FlashAttention only supports Ampere GPUs or newer` on V100s.

**Fix**: Always request H100 nodes: `srun --constraint="h100"`. Without flash attention, the model falls back to PyTorch SDPA automatically (no code change needed — just uninstall flash-attn).

### 5. Flash attention build issues on RHEL 8

See `INSTALLATION.md` for the full list (CUDA_HOME, psutil, NFS cross-device link, GLIBC 2.32).

## Sampling CLI reference

```
python sample.py --checkpoint PATH [options]

Required:
  --checkpoint PATH       Path to trained .ckpt file

Options:
  --num-samples N         Number of sequences (default: 10, ignored with --use-test-labels)
  --steps N               Sampling steps (default: 128)
  --batch-size N          GPU batch size (default: 64)
  --output-dir DIR        Output directory (default: generated/)
  --use-test-labels       Use test set labels for conditional generation
  --config PATH           Config YAML (default: config.yaml)
```

## Performance (H100 NVL, 94 GB, promoter 1024bp, 128 steps)

Throughput is **constant at ~5.9 seq/s** regardless of batch size. The model is compute-bound on H100 — larger batches don't improve throughput, they just use more memory.

| Batch size | Peak memory | Time (128 samples) | Throughput |
|---|---|---|---|
| 64 | 2.3 GB | 10.9s | 5.9 seq/s |
| 128 | 4.0 GB | 21.5s | 5.9 seq/s |
| 256 | 7.5 GB | 43.4s | 5.9 seq/s |
| 512 | 14.3 GB | — | 5.9 seq/s |
| 1024 | 28.0 GB | — | 5.8 seq/s |
| 1536 | 41.7 GB | — | 5.9 seq/s |

Memory scales linearly at ~27 MB/sample. The H100 NVL (94 GB) can fit ~3400 samples in a single batch, but flash_attn 2.5.8 becomes unstable above ~2048 (illegal memory access errors). **Recommended batch size: 512** — keeps memory reasonable and avoids flash_attn edge cases.

**Time estimates for the full promoter test set (7497 samples, 128 steps):**
- Wall time: 7497 / 5.9 ≈ **~21 minutes** (independent of batch size)
- Request `--time=00:30:00` for SLURM jobs

## Output formats

- **NPZ**: One-hot encoded, shape `(N, seq_len, 4)`, key `arr_0`
- **FASTA**: Standard FASTA with headers `>seq_0`, `>seq_1`, etc.
