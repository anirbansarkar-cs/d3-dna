# Promoter example

End-to-end D3 conditional diffusion on the FANTOM5 promoter dataset: **1024 bp sequences with per-position CAGE activity labels** (shape `(N, 1024, 1)`). Two architectures are supported out of the box: a transformer (12-block DDiT) and a 20-block dilated convolutional model.

## Prerequisites

- `d3-dna` installed from source (the package isn't on PyPI yet).
- GPU with Ampere architecture or newer if `flash-attn` is installed (H100 recommended); otherwise any CUDA GPU — the transformer falls back to PyTorch SDPA automatically.
- `curl` on PATH (used to fetch defaults from Zenodo).

**Data + oracle weights auto-download from [Zenodo record 19738941](https://zenodo.org/records/19738941) on first run** and cache under `examples/promoter/cache/` (gitignored). To use a pre-existing copy on a shared filesystem instead, pass `--data-file /path/to/Promoter_data.npz` and `--oracle-file /path/to/oracle.pth.tar` to any of the scripts. Using the SEI oracle requires a feature mask (also on Zenodo and automatically downloaded.)

## Files

| File | Description |
|---|---|
| `config_transformer.yaml` | Transformer config (default). |
| `config_conv.yaml` | Convolutional config. |
| `data.py` | `PromoterDataset` — loads the NPZ and yields `(X: LongTensor[1024], y: FloatTensor[1024, 1])`. |
| `oracle.py` | SEI oracle loader (B-spline basis, strand averaging, H3K4me3 masking). |
| `callbacks.py` | `PromoterSPMSECallback` — periodic SP-MSE validation against SEI during training. |
| `train.py` | Training script using `D3Trainer`. |
| `sample.py` | Conditional sampling using `D3Sampler`. |
| `evaluate.py` | Evaluation via `D3Evaluator` (MSE, KS, JS, AUROC) + SEI oracle. |

## Usage

### 1. Train

```bash
# Transformer
python train.py --config config_transformer.yaml

# Convolutional
python train.py --config config_conv.yaml --output-dir outputs/promoter_conv

# Resume
python train.py --config config_transformer.yaml \
    --resume-from outputs/promoter_transformer/checkpoints/last.ckpt
```

Checkpoints land in `outputs/promoter_{architecture}/checkpoints/`. `PromoterSPMSECallback` tracks SP-MSE periodically during training.

### 2. Sample

```bash
# 10 random-label samples, 128 steps (defaults)
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt

# One sample per test-set TSS
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt --use-test-labels

# n-per-TSS protocol (averaging over n samples)
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt \
    --use-test-labels --paired-repeat n
```

Output: `generated/samples.npz` (one-hot `(N, 1024, 4)`) and `generated/samples.fasta`.

### 3. Evaluate

```bash
# Full evaluation (MSE, KS, JS, AUROC). Default JS is single-k at k=6.
python evaluate.py --samples generated/samples.npz

# Report JS distance averaged over k ∈ {1..7}
python evaluate.py --samples generated/samples.npz --kmer-ks 1-7

# DDSM 5-per-TSS paired evaluation
python evaluate.py --samples generated/samples.npz --paired-repeat 5

# Subset of metrics
python evaluate.py --samples-dir generated --tests mse,ks
```

`evaluate.py` loads the SEI oracle (for MSE/KS), reads the real-data NPZ (one-hot channels 0–3), and dispatches through `D3Evaluator`.

```bash
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
    --job-name=d3_promoter_eval --wrap="source /path/to/conda.sh && conda activate <env> && \
        python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt --use-test-labels && \
        python evaluate.py --samples generated/samples.npz"
```

## Per-position vs global conditioning

Unlike K562/LentiMPRA (single scalar per sequence), promoter labels are **per-position** with shape `(batch, 1024, 1)`. The core `EmbeddingLayer` auto-detects the 3D label tensor and adds it element-wise rather than broadcasting. No config flag controls this — it's determined by the label tensor dimensionality.

## Floating-point precision

> **Promoter exception.** The package default is `bf16-mixed` for transformer
> architectures and `16-mixed` (fp16 + GradScaler) for convolutional. Promoter
> overrides the transformer side back to `16-mixed` via `cfg.training.precision`
> in `config_transformer.yaml` because the Zenodo `D3_Tran_Promoter.ckpt` was
> trained / validated under fp16; sampling it in bf16 produces materially
> degraded outputs (we observed AUROC ≈ 1.0 vs ≈ 0.55 in fp16). This exception
> is local to promoter — k562, deepstarr, hepg2 etc. should keep the
> architecture-driven default.

The d3-dna core pipeline is identical across every example in this repo — the precision story below applies to the promoter example in particular but also matches k562, hepg2, deepstarr, and minimal unchanged.

**Training (CUDA).** Lightning runs with `precision='16-mixed'`, which installs a `GradScaler` and wraps training in `torch.amp.autocast(dtype=torch.float16)` (`d3_dna/modules/trainer.py:454`). `get_score_fn` adds a second redundant autocast block around every model forward so the same policy applies whether the forward is driven by the training loss or the sampler (`d3_dna/models/diffusion.py:446`). Matmul/conv ops in DDiT / conv blocks run in fp16; the custom `LayerNorm` (`d3_dna/models/layers.py:138–142`) explicitly opts out with `autocast(enabled=False)` and does `F.layer_norm(x.float(), ...)` in fp32 — a stability fix for fp16 variance. TF32 matmul is enabled globally via `torch.set_float32_matmul_precision('medium')` (`d3_dna/modules/trainer.py:27`). Score-entropy and other loss-side reductions end up in fp32 via autocast's fp32-op list (`exp`, `log`, `sum`, `softmax`, …).

**Sampling (CUDA).** Every model forward during sampling goes through the *same* autocast block (`get_score_fn(..., sampling=True)` at `d3_dna/models/diffusion.py:633`). Matmul/conv run fp16, LayerNorm still fp32. The score itself is **not** explicitly cast anywhere in the predictor / graph path. However two implicit upcasts land the sampler arithmetic back in fp32:

1. `score.exp()` at `d3_dna/models/diffusion.py:455` runs under autocast; `torch.exp` is on PyTorch's fp32-cast op list, so the returned tensor is fp32 regardless of the model's output dtype.
2. `sigma`, `dsigma`, `step_size`, and the rate tensors built via `torch.ones(...)` in `Uniform`/`Absorbing` (e.g. `d3_dna/models/diffusion.py:179,187,192`) are created in the default fp32 dtype. PyTorch type promotion then forces the predictor's `score * dsigma * …` arithmetic into fp32.

Net effect: the model's *inner activations* are fp16-mixed; the *score and all post-model sampler state* are fp32. Weights and EMA shadow are stored fp32. On CPU the `enabled=(device_type == 'cuda')` guard turns the autocast off entirely, so CPU sampling is fp32 end-to-end with no mixed-precision at all.

**Evaluation.** Never touches the diffusion model — only the oracle. Inputs pass through `D3Evaluator._normalize_onehot` which casts to `np.float32`; SEI (`oracle.py`) and the discriminability CNN (`d3_dna/evals/metrics.py`) forward in fp32. No autocast on this path.

| Stage | What ends up in fp16 vs fp32 |
|---|---|
| Training | Matmul/conv fp16; LayerNorm fp32 island; loss-side reductions fp32 (autocast fp32-op list). GradScaler for fp16 gradient stability. |
| Sampling (CUDA) | Matmul/conv fp16; LayerNorm fp32 island; **score returned as fp32** (implicit upcast via `.exp()` autocast rule); predictor arithmetic fp32 (type promotion against default-fp32 sigma/rate tensors). No explicit `.float()` on the score. |
| Sampling (CPU) | Everything fp32 — autocast disabled by the `is_cuda` guard. |
| Evaluation | fp32 throughout. |

## Metrics

| Metric | Description | Direction |
|---|---|---|
| `fidelity_mse` | Paired MSE of SEI oracle predictions (real vs generated) | Lower is better |
| `ks_statistic` | Mean per-feature two-sample KS on oracle predictions | Lower is better |
| `js_distance` | Jensen-Shannon **distance** (sqrt of divergence; satisfies the triangle inequality) between k-mer distributions. Default: single k=6. With `--kmer-ks 1-7` (or any interval/list), returns the mean over those k's. | Lower is better |
| `auroc` | AUROC of a CNN discriminator (real=1, gen=0) | Closer to 0.5 is better |

## Reference results

Run end-to-end from the public Zenodo artifacts (`D3_Tran_Promoter.ckpt`, `D3_Conv_Promoter.ckpt`, `data_Promoter.npz`, `Oracle_Promoter.pth.tar`) on a single H100 NVL. Sampling: one sequence per TSS in the FANTOM5 test split (`--use-test-labels`, `paired_repeat=1`, **100 PC steps**). JS reported at single `k=6`. Both rows reflect the precision policy actually used to produce the public checkpoints — see "Floating-point precision" above.

| Architecture | Precision (train / sample) | `fidelity_mse` ↓ | `ks_statistic` ↓ | `js_distance` (k=6) ↓ | `auroc` (→ 0.5) |
|---|---|---|---|---|---|
| Transformer (12-block DDiT, ~30 M) | fp16 / fp16 | 0.027365 | 0.045885 | 0.029593 | 0.560604 |
| Convolutional (20 dilated blocks, 256 ch) | fp16 / fp16 | 0.027531 | 0.067360 | 0.024207 | 0.553784 |

Reproduce with:

```bash
python sample.py   --config config_transformer.yaml --use-test-labels --steps 100
python evaluate.py --config config_transformer.yaml --samples-dir generated --kmer-ks 6
# (and the same with --config config_conv.yaml)
```
