# Promoter example

End-to-end D3 conditional diffusion on the FANTOM5 promoter dataset: **1024 bp sequences with per-position CAGE activity labels** (shape `(N, 1024, 1)`). Two architectures are supported out of the box: a transformer (12-block DDiT, ~30M params) and a 20-block dilated convolutional model.

> **Loading a pretrained hybrid SEDD checkpoint?** See [`legacy/`](legacy/). That path is sampling-only and isolated from the from-scratch training flow described here.

## Prerequisites

- `d3-dna` installed from source (the package isn't on PyPI yet).
- GPU with Ampere architecture or newer if `flash-attn` is installed (H100 recommended); otherwise any CUDA GPU — the transformer falls back to PyTorch SDPA automatically.
- Promoter NPZ at `paths.data_file` (default: `/grid/koo/home/shared/d3/data/promoter/Promoter_data.npz`).
- SEI oracle checkpoint at `paths.oracle_model` (default: `/grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar`).

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
# Transformer (default)
python train.py

# Convolutional
python train.py --config config_conv.yaml --work-dir outputs/promoter_conv

# Resume
python train.py --config config_transformer.yaml \
    --resume outputs/promoter_transformer/checkpoints/last.ckpt
```

Checkpoints land in `outputs/promoter_{architecture}/checkpoints/`. `PromoterSPMSECallback` tracks SP-MSE periodically during training.

Submit on SLURM (replace `<env>` and the conda init):

```bash
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --job-name=d3_promoter_train --wrap="source /path/to/conda.sh && conda activate <env> && python train.py"
```

### 2. Sample

```bash
# 10 random-label samples, 128 steps (defaults)
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt

# One sample per test-set TSS
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt --use-test-labels

# DDSM 5-per-TSS protocol
python sample.py --checkpoint outputs/promoter_transformer/checkpoints/last.ckpt \
    --use-test-labels --paired-repeat 5
```

Output: `generated/samples.npz` (one-hot `(N, 1024, 4)`) and `generated/samples.fasta`.

### 3. Evaluate

```bash
# Full evaluation (MSE, KS, JS, AUROC). Default JS is single-k at k=6.
python evaluate.py --samples generated/samples.npz

# Report JS divergence averaged over k ∈ {1..7}
python evaluate.py --samples generated/samples.npz --kmer-ks 1-7

# DDSM 5-per-TSS paired evaluation
python evaluate.py --samples generated/samples.npz --paired-repeat 5

# Subset of metrics
python evaluate.py --samples-dir generated --tests mse,ks

# JS averaged over k ∈ {1..7} instead of single k=6
python evaluate.py --samples-dir generated --kmer-ks 1-7
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
| `js_divergence` | JS divergence of k-mer distributions. Default: single k=6. With `--kmer-ks 1-7` (or any interval/list), returns the mean over those k's. | Lower is better |
| `auroc` | AUROC of a CNN discriminator (real=1, gen=0) | Closer to 0.5 is better |
