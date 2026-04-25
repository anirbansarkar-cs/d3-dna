# K562 LentiMPRA example

End-to-end D3 conditional diffusion on the K562 LentiMPRA dataset: **230 bp regulatory sequences with a single global regression target** (shape `(N, 1)`) — one activity measurement per sequence, broadcast across all positions by the embedding layer.

## Prerequisites

- `d3-dna` installed from source.
- GPU with Ampere architecture or newer if `flash-attn` is installed; otherwise any CUDA GPU (SDPA fallback).
- K562 LentiMPRA H5 at `paths.data_file` (default: `data/lenti_MPRA_K562_data.h5`).
- LegNet oracle checkpoint at `paths.oracle_model` (default: `data/oracle_best_model.ckpt`).
- `config.yaml` requests `ngpus: 2` out of the box; reduce to 1 if running on a single GPU.

## Files

| File | Description |
|---|---|
| `config.yaml` | Model/training/optim config (12-block transformer, ~30M params). |
| `data.py` | `K562Dataset` — loads H5, exposes train/valid/test splits. |
| `oracle.py` | Vendored LegNet oracle (no `mpralegnet` install needed). |
| `train.py` | Training script using `D3Trainer`. |
| `sample.py` | Conditional sampling (N replicates) using `D3Sampler`. |
| `evaluate.py` | Built-in SP-MSE over replicates, plus an optional hand-off to the external `d3_evaluation_pipeline` for the full 5-metric suite. |

## Usage

### 1. Train

```bash
python train.py
python train.py --resume outputs/k562/checkpoints/model-epoch=100-val_loss=260.ckpt
```

Checkpoints land in `outputs/k562/checkpoints/`. Best checkpoint is the one with the lowest `val_loss` in its filename.

Submit on SLURM (2 GPUs by default per `config.yaml`):

```bash
sbatch --partition=gpu --gres=gpu:2 --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --job-name=d3_k562_train --wrap="source /path/to/conda.sh && conda activate <env> && python train.py"
```

### 2. Sample

Defaults: `--steps 20`, `--replicates 5`, `--batch-size` = training batch size from config, checkpoint auto-selected (lowest `val_loss` among `outputs/k562/checkpoints/model-*.ckpt`).

```bash
# Default: 20 steps × 5 replicates × len(test set)
python sample.py

# Higher-fidelity sampling (more steps, explicit checkpoint)
python sample.py --checkpoint outputs/k562/checkpoints/model-epoch=263.ckpt --steps 230

# Single replicate with offset (for parallel SLURM array jobs)
python sample.py --replicates 1 --rep-offset 3 --steps 200
```

Output: `generated/sample_{i}.npz` (one-hot `(N, 230, 4)`) and `generated/sample_{i}.fasta` per replicate.

### 3. Evaluate

```bash
# Built-in SP-MSE across all sample_*.npz replicates
python evaluate.py \
    --samples-dir generated \
    --data data/lenti_MPRA_K562_data.h5 \
    --oracle data/oracle_best_model.ckpt

# Full 5-metric suite via the external pipeline
python evaluate.py \
    --samples-dir generated \
    --data data/lenti_MPRA_K562_data.h5 \
    --oracle data/oracle_best_model.ckpt \
    --eval-pipeline /path/to/d3_evaluation_pipeline
```

Note: this example's `evaluate.py` implements SP-MSE in-place (does **not** go through `D3Evaluator`). To run the four-metric `D3Evaluator` suite (`mse`, `ks`, `js`, `auroc`) on K562 samples, see `examples/promoter/evaluate.py` as a reference and adapt the oracle loader.

Chained sample + evaluate on SLURM:

```bash
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
    --job-name=d3_k562_eval --wrap="source /path/to/conda.sh && conda activate <env> && \
        python sample.py --steps 20 --replicates 5 && \
        python evaluate.py --samples-dir generated \
            --data data/lenti_MPRA_K562_data.h5 --oracle data/oracle_best_model.ckpt"
```

## Floating-point precision

The d3-dna core is shared, so K562's precision behaviour is identical to the other examples. See [`examples/promoter/README.md`](../promoter/README.md#floating-point-precision) for the full explanation (training autocast + GradScaler, LayerNorm fp32 island, implicit fp32 upcast of the score via `exp` + type promotion against sigma/rate tensors, fp32 evaluation through the oracle). K562-specific details:

- **Evaluation** in this example is the built-in in-place SP-MSE path (not `D3Evaluator`), so the numpy cast is at `evaluate.py:63,76` (`.astype(np.float32)`) and the LegNet oracle's `.predict()` re-casts its input to `torch.float32` (`oracle.py`). No autocast is active during evaluation — it's fp32 end to end.
- **Training** defaults to `ngpus: 2` in `config.yaml`; DDP and fp16-mixed autocast compose without changes.

| Stage | What ends up in fp16 vs fp32 |
|---|---|
| Training | Matmul/conv fp16; LayerNorm fp32 island; loss-side reductions fp32 (autocast fp32-op list). GradScaler for fp16 gradient stability. |
| Sampling (CUDA) | Matmul/conv fp16; LayerNorm fp32 island; score returned as fp32 (implicit upcast via `.exp()` autocast rule); predictor arithmetic fp32 (type promotion against default-fp32 sigma/rate tensors). No explicit `.float()` on the score. |
| Sampling (CPU) | fp32 throughout — autocast disabled by the `is_cuda` guard. |
| Evaluation | fp32 throughout (LegNet oracle forward). |

## Metrics

Built-in (`evaluate.py`):

| Metric | Description | Direction |
|---|---|---|
| SP-MSE | Paired MSE of LegNet oracle predictions (test vs each replicate) | Lower is better |

Via `--eval-pipeline` (external `d3_evaluation_pipeline`):

| Metric | Description | Direction |
|---|---|---|
| Discriminability AUROC | Real vs generated CNN classifier | Closer to 0.5 is better |
| Predictive-dist shift | KS on oracle predictions | Lower is better |
| K-mer spectrum shift | JS on k-mer frequencies | Lower is better |
| Conditional-generation fidelity | Oracle-prediction MSE | Lower is better |
| Percent identity | Max pairwise identity to training set | Reference |

## Reference results

Mean across replicates from a separate evaluation pipeline. JS reported at single `k=6`. Lower is better on every metric except AUROC, which targets 0.5.

| Architecture | `fidelity_mse` ↓ | `ks_statistic` ↓ | `js_distance` (k=6) ↓ | `auroc` (→ 0.5) |
|---|---|---|---|---|
| Convolutional | 0.2185 | 0.05942 | 0.03671 | 0.7700 |
| Transformer | 0.1994 | 0.00827 | 0.02684 | 0.7248 |
