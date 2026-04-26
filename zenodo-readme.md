# D3-DNA: Data and Model Weights

Training data, oracle regressors, and pretrained diffusion-model checkpoints for **D3 (DNA Discrete Diffusion)** — a conditional discrete-diffusion framework for regulatory DNA generation. Four datasets: Promoter, DeepSTARR, MPRA-K562, MPRA-HepG2.

- **Code (full):** https://github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion
- **Code (minimal package):** https://github.com/anirbansarkar-cs/d3-dna
- **License:** CC BY 4.0

## Files

### Promoter — 1024 bp, per-position H3K4me3 signal

| File | Size | MD5 | Description |
|---|---:|---|---|
| `data_Promoter.npz` | 246.0 MB | `ebd9ff5a884a30cc63ae645856063e76` | Train/valid/test, `(N, 1024, 6)`: channels 0–3 one-hot, channel 4 H3K4me3 signal (label) |
| `Oracle_Promoter.pth.tar` | 7.12 GB | `e43843b40c904956f39fc6edd48055ac` | Sei multi-task trained weights (Chen et al.)|
| `target.sei.names` | 905 KB | — | Sei feature-name file; selects the 2350 H3K4me3 tracks from Sei's 21907 outputs |
| `D3_Tran_Promoter.ckpt` | 1.48 GB | `de40be87592cc77cb8ecf4e269dabac9` | Transformer backbone |
| `D3_Conv_Promoter.ckpt` | 201.9 MB | `c4c74a36daa44e63b4a616568e1995b4` | Convolutional backbone |

### DeepSTARR — 249 bp Drosophila enhancers, dual-head (Dev / Hk)

| File | Size | MD5 | Description |
|---|---:|---|---|
| `data_DeepSTARR.h5` | 294.6 MB | `f9d0cae9ce4dd3c90b465e407d842cc7` | Train/valid/test one-hot + dual (dev, hk gene schedules) MPRA activity |
| `Oracle_DeepSTARR.ckpt` | 7.5 MB | `d9b50c4f358e8bef72e1eff8d3951630` | DeepSTARR CNN trained weights (de Almeida et al.) |
| `D3_Tran_DeepSTARR.ckpt` | 1.48 GB | `fb883c4f3114dc6b4757c546d9642a44` | Transformer backbone |
| `D3_Conv_DeepSTARR.pth` | 214.6 MB | `d1243fb01ed24b69a3643a03ff9b6680` | Convolutional backbone |

### MPRA-K562 — 230 bp lentiMPRA, single global scalar

| File | Size | MD5 | Description |
|---|---:|---|---|
| `data_MPRA_K562.h5` | 1.55 GB | `da0df7173e2e00aad4e37c8995af7ee4` | Train/valid/test one-hot + single scalar MPRA activity |
| `Oracle_MPRA_K562.ckpt` | 16.0 MB | `1b7b52d9416bb88d4eb8f5740b8d1d6f` | LegNet architecture (Penzar et al.), weights trained on our own|
| `D3_Tran_MPRA_K562.ckpt` | 1.48 GB | `e282c76933471a005f4ab821053ce4ee` | Transformer backbone |
| `D3_Conv_MPRA_K562.ckpt` | 214.1 MB | `23bd19ea598805d0ac1ef2cd3d414f2c` | Convolutional backbone |

### MPRA-HepG2 — 230 bp lentiMPRA, single global scalar

| File | Size | MD5 | Description |
|---|---:|---|---|
| `data_MPRA_HepG2.h5` | 970.7 MB | `b37762d49dac75e4036c02398b468c8b` | Train/valid/test one-hot + scalar regression target |
| `Oracle_MPRA_HepG2.ckpt` | 16.0 MB | `37ca20c28d8f99ac72401f784bf0869c` | LegNet architecture (Penzar et al.), weights trained on our own|
| `D3_Tran_MPRA_HepG2.ckpt` | 1.48 GB | `f8f13d2f8145478fe322587eed5d1e15` | Transformer backbone |
| `D3_Conv_MPRA_HepG2.pth` | 214.5 MB | `963c6b25c3e249bc99af1215084d4e3c` | Convolutional backbone |

## Data Loading

### H5 format

```python
import h5py
with h5py.File('data_MPRA_K562.h5', 'r') as f:
    onehot_train = f['onehot_train'][()]   # (N, 230, 4)
    y_train      = f['y_train'][()]        # (N, 1)
    onehot_test  = f['onehot_test'][()]    # (N, 230, 4)
    y_test       = f['y_test'][()]         # (N, 1)
```

### NPZ compressed format

```python
import numpy as np
d = np.load('data_Promoter.npz')
# 'train' / 'valid' / 'test' — each (N, 1024, 6): channels [0:4]=one-hot, [4:6]=H3K4me3 signal
train = d['train']
```


## Notes

**Mixed precision.** Default policy in `d3-dna`: transformer → `bf16-mixed`, convolutional → `16-mixed` (fp16 + GradScaler). Both training and sampling share one autocast policy via `get_score_fn`. **Promoter is an exception** — both backbones use fp16. Override per config via `cfg.training.precision`.

**Checkpoint formats.** Both PyTorch Lightning `.ckpt` and original SEDD `.pth` (top-level `model` / `ema` / `step`) are accepted by `d3_dna.modules.checkpoint.load_checkpoint`; dispatch is by file content, not extension.

## Contact

Peter Koo — `koo@cshl.edu`
