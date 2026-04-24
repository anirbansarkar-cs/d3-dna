# Minimal example

Smallest possible D3 training loop: a synthetic dataset of random 64-bp sequences with scalar labels, trained for 2 epochs on a small transformer. Use this as a quickstart smoke test or a template for new datasets.

## Files

| File | Description |
|---|---|
| `config.yaml` | Small transformer config (4 blocks, hidden 128), 2-epoch training. |
| `data.py` | `make_synthetic_dataset(n_samples, seq_len)` — returns a `TensorDataset` of (tokens, scalar labels). |
| `train.py` | Builds the dataset, calls `D3Trainer.fit`. |
| `sample.py` | Loads the newest checkpoint, generates 10 sequences, writes `generated.fasta`. |

## Usage

```bash
# Train (≈ a couple of minutes on one GPU)
python train.py

# Sample 10 sequences from the freshest checkpoint
python sample.py
```

SLURM one-shot:

```bash
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 \
    --job-name=d3_minimal --wrap="source /path/to/conda.sh && conda activate <env> && \
        python train.py && python sample.py"
```

## Floating-point precision

Identical to the other examples — see [`examples/promoter/README.md`](../promoter/README.md#floating-point-precision) for the full explanation (fp16-mixed autocast on CUDA for matmul/conv, LayerNorm fp32 island, score implicitly upcast to fp32 after `.exp()`, predictor arithmetic fp32, fp32 on CPU).
