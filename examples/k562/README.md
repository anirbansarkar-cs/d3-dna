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
| `train.py` | Training script using `D3Trainer` |
| `sample.py` | Conditional sampling using `D3Sampler` |
| `evaluate.py` | Evaluation: built-in SP-MSE + optional external 5-metric pipeline |

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
# Built-in SP-MSE only
python evaluate.py \
    --samples-dir generated \
    --data data/lenti_MPRA_K562_data.h5 \
    --oracle data/oracle_best_model.ckpt

# Full 5-metric evaluation (requires external pipeline)
python evaluate.py \
    --samples-dir generated \
    --data data/lenti_MPRA_K562_data.h5 \
    --oracle data/oracle_best_model.ckpt \
    --eval-pipeline /path/to/d3_evaluation_pipeline
```

### Oracle dependency

The evaluation script requires the [mpralegnet](https://github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion) LegNet oracle package to be installed or available on `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/mpralegnet:$PYTHONPATH"
```

This is needed for both the built-in SP-MSE computation and the external evaluation pipeline.

## Metrics

| Metric | Description | Direction |
|---|---|---|
| SP-MSE | Sample-Prediction MSE (oracle fidelity) | Lower is better |
| Discriminability AUROC | Real vs generated classification | Higher is better |
| K-mer JS distance | K-mer spectrum divergence | Lower is better |
| Percent identity | Max pairwise identity to training set | Similarity measure |
| Predictive dist. shift | KS statistic on oracle predictions | Lower is better |
