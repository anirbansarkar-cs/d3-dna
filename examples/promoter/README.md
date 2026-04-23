# Promoter Example

End-to-end example of training, sampling, and evaluating a D3 conditional diffusion model on the FANTOM5 promoter dataset (1024 bp sequences, per-position CAGE activity signal). Two architectures are supported out of the box: a transformer (DDiT, 12 blocks) and a 20-block dilated convolutional model.

> **Loading a pretrained hybrid SEDD checkpoint?** See [`legacy/`](legacy/). That path is sampling-only and isolated from the from-scratch training flow described here.

## Prerequisites

- `d3-dna` package installed (see [main README](../../README.md))
- GPU with Ampere architecture or newer (required for flash attention)
- Promoter data file (`Promoter_data.npz`) — the default path in the config is `/grid/koo/home/shared/d3/data/promoter/Promoter_data.npz`
- SEI oracle checkpoint for SP-MSE validation and evaluation — default `/grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar`

## Files

| File | Description |
|---|---|
| `config_transformer.yaml` | Transformer architecture (12-block DDiT, ~30M params). Default. |
| `config_conv.yaml` | Convolutional architecture (20 dilated conv blocks, 256 channels). |
| `data.py` | `PromoterDataset` class: loads NPZ, yields `(X: LongTensor[1024], y: FloatTensor[1024, 1])`. |
| `oracle.py` | SEI oracle model (B-spline basis, strand averaging, H3K4me3 masking). |
| `callbacks.py` | `PromoterSPMSECallback` — SP-MSE validation against SEI during training. |
| `train.py` | Training script using `D3Trainer`. |
| `sample.py` | Conditional sampling using `D3Sampler`. |
| `evaluate.py` | Evaluation: MSE, KS, JS, AUROC via `D3Evaluator` + SEI oracle. |

## Usage

### 1. Training

```bash
# Transformer (default)
python train.py

# Convolutional
python train.py --config config_conv.yaml --work-dir outputs/promoter_conv

# Resume
python train.py --resume outputs/promoter_transformer/checkpoints/last.ckpt
```

Checkpoints are saved to `outputs/promoter_{architecture}/checkpoints/`. `D3Trainer` monitors validation loss and SP-MSE (via `PromoterSPMSECallback`).

**Per-position conditioning.** Unlike K562/LentiMPRA where labels are a single scalar per sequence, promoter labels are per-position — shape `(batch, 1024, 1)`. The core `EmbeddingLayer` auto-detects the 3D label tensor and broadcasts it across the sequence.

### 2. Sampling

```bash
# 10 random-label samples, 128 steps, transformer config (default)
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt

# Generate one sample per test-set TSS
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt --use-test-labels

# DDSM 5-per-TSS protocol
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt --use-test-labels --paired-repeat 5
```

Output: `generated/samples.npz` (one-hot `(N, 1024, 4)`) and `generated/samples.fasta`.

### 3. Evaluation

```bash
python evaluate.py --samples generated/samples.npz

# DDSM 5-per-TSS protocol: tile real set 5x before pairing
python evaluate.py --samples generated/samples.npz --paired-repeat 5

# Subset of metrics
python evaluate.py --samples generated/samples.npz --tests mse,ks
```

`evaluate.py` loads the SEI oracle (for MSE/KS), extracts the one-hot DNA channels from the 40k NPZ, and dispatches metric computation through `D3Evaluator`.

## Metrics

| Metric | Description | Direction |
|---|---|---|
| SP-MSE (`mse`) | Oracle-prediction MSE (sample vs real) | Lower is better |
| KS (`ks`) | KS statistic on oracle predictions | Lower is better |
| JS divergence (`js`) | k-mer spectrum JS divergence | Lower is better |
| AUROC (`auroc`) | Real vs generated discriminability CNN | Lower is better (closer to 0.5) |
