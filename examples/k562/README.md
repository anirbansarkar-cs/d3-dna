# K562 LentiMPRA Example

End-to-end example of training, sampling, and evaluating a D3 conditional diffusion model on the K562 LentiMPRA dataset (230 bp regulatory sequences, single regression target).

## Prerequisites

- `d3-dna` package installed (see [main README](../../README.md))
- GPU with Ampere architecture or newer (required for flash attention)
- K562 LentiMPRA data file (`lenti_MPRA_K562_data.h5`)
- LegNet oracle checkpoint for evaluation

## Files

| File | Description |
|---|---|
| `config.yaml` | Model and training configuration |
| `data.py` | `K562Dataset` class (loads H5, splits train/valid/test) |
| `oracle.py` | Vendored LegNet oracle (no external package). |
| `callbacks.py` | `K562MSECallback` — SP-MSE validation against LegNet during training. |
| `train.py` | Training script using `D3Trainer` with `K562MSECallback` wired in. |
| `sample.py` | Conditional sampling using `D3Sampler` (produces `sample_*.npz` replicates). |
| `evaluate.py` | MSE, KS, JS, AUROC via `D3Evaluator` + LegNet oracle, per-replicate + mean. |

## Usage

### 1. Training

```bash
python train.py
```

Checkpoints are saved to `outputs/k562/checkpoints/`. The best checkpoint is selected by lowest `val_loss`.

### 2. Sampling

```bash
# Default: 20 steps, 5 replicates, auto-selects best checkpoint
python sample.py

# Custom: 200 steps, specific checkpoint
python sample.py --checkpoint outputs/k562/checkpoints/model-epoch=123-val_loss=247.05.ckpt --steps 200

# Single replicate with offset (for parallel SLURM array jobs)
python sample.py --replicates 1 --rep-offset 3 --steps 200
```

Output: `generated/sample_{i}.npz` (one-hot) and `generated/sample_{i}.fasta` per replicate.

### 3. Evaluation

```bash
# All four metrics, one row per sample_*.npz plus a mean row
python evaluate.py \
    --samples-dir generated \
    --data data/lenti_MPRA_K562_data.h5 \
    --oracle data/oracle_best_model.ckpt

# Subset of metrics
python evaluate.py --samples-dir generated --data <H5> --oracle <ckpt> --tests mse,ks

# JS averaged over k ∈ {1..7} instead of single k=6
python evaluate.py --samples-dir generated --data <H5> --oracle <ckpt> --kmer-ks 1-7
```

Output: `eval_results/<replicate>.json` per sample file, `eval_results/summary.csv` (per-replicate rows + mean), `eval_results/summary.json` (full structure).

### Oracle dependency

The LegNet oracle architecture is vendored in `oracle.py`. `evaluate.py` needs the checkpoint (`--oracle`) and optionally a LegNet config (`--oracle-config`, defaults to the shared `DEFAULT_CONFIG` path). No external evaluation pipeline required.

## Metrics

| Metric | Description | Direction |
|---|---|---|
| `fidelity_mse` | Paired MSE of LegNet oracle predictions (real vs generated) | Lower is better |
| `ks_statistic` | Mean per-feature two-sample KS on oracle predictions | Lower is better |
| `js_divergence` | JS divergence of k-mer distributions. Default: single k=6. With `--kmer-ks 1-7` (or any interval/list), returns the mean over those k's. | Lower is better |
| `auroc` | AUROC of a CNN discriminator (real=1, gen=0) | Closer to 0.5 is better |
