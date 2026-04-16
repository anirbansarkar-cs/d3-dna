# d3-dna Evaluation Guide

Tested on: RHEL 8, SLURM cluster with H100 NVL GPU, conda env `d3-dna`, 2026-04-09.

## Quick start

```bash
srun --partition=gpuq --gres=gpu:1 --constraint="h100" --time=01:00:00 bash -c '
  conda activate d3-dna
  cd d3-dna/examples/promoter
  python evaluate.py \
    --samples generated_test/samples.npz \
    --data /grid/koo/home/shared/d3/data/promoter/Promoter_data.npz \
    --model /grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar
'
```

## Metrics

| Metric | Name in script | What it measures | Needs oracle |
|---|---|---|---|
| **MSE** | `mse` | Mean squared error between SEI oracle predictions on test vs generated sequences. Lower = generated sequences have more similar functional profiles to real ones. | Yes |
| **KS statistic** | `ks_statistic` | Kolmogorov-Smirnov two-sample test on the distribution of mean oracle predictions. Measures whether the overall distribution of predicted activity is shifted. Range [0, 1], lower = more similar distributions. | Yes |
| **JS distance** | `js_distance` | Jensen-Shannon divergence between 6-mer frequency distributions of test and generated sequences. Measures compositional similarity at the k-mer level. Range [0, 1], lower = more similar k-mer profiles. | No |
| **AUROC** | `auroc` | AUROC of a CNN binary classifier trained to distinguish real from synthetic sequences. 0.5 = indistinguishable (ideal), 1.0 = trivially distinguishable. | No |

## Input formats

### Samples NPZ
Generated sequences in one-hot format, saved by `sample.py`:
- Key: `arr_0`
- Shape: `(N, seq_len, 4)` — e.g., `(7497, 1024, 4)`
- The script transposes to `(N, 4, seq_len)` internally

### Promoter data NPZ
Training/test data at `/grid/koo/home/shared/d3/data/promoter/Promoter_data.npz`:
- Keys: `train`, `test`, `valid`
- Shape per split: `(N, 1024, 6)` — channels 0-3 are one-hot DNA, channels 4-5 are activity
- The script extracts channels 0-3 only

### SEI oracle model
Located at `/grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar`:
- Architecture: Deep CNN with residual and dilated convolutions
- Input: `(batch, 4, 4096)` — sequences are padded to 4096bp with uniform 0.25 background
- Output: `(batch, 21907)` — predictions for 21,907 genomic features
- Wrapped in `NonStrandSpecific` — averages forward and reverse complement predictions

## SEI padding

SEI requires 4096bp input but promoter sequences are 1024bp. The evaluation script pads symmetrically with uniform background (0.25 per nucleotide):

```
[0.25 padding (1536bp)] [sequence (1024bp)] [0.25 padding (1536bp)] = 4096bp
```

This matches how the SEI model was designed to handle shorter sequences.

## CLI reference

```
python evaluate.py --samples PATH [options]

Required:
  --samples PATH          Path to generated samples NPZ

Options:
  --data PATH             Promoter data NPZ (default: /grid/koo/home/shared/d3/data/promoter/Promoter_data.npz)
  --model PATH            SEI oracle checkpoint (default: /grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar)
  --test TESTS            Comma-separated list of tests (default: all)
                          Options: mse, ks_statistic, js_distance, auroc
  --output-dir DIR        Output directory (default: eval_results/)
  --kmer-k K              k-mer length for JS distance (default: 6)
```

## Running specific tests

```bash
# Oracle-dependent tests only (faster if you just need functional metrics)
python evaluate.py --samples samples.npz --test "mse,ks_statistic"

# Sequence-level tests only (no oracle needed, much faster)
python evaluate.py --samples samples.npz --test "js_distance,auroc"
```

## Performance

On H100 NVL with 7497 test sequences:

| Phase | Time |
|---|---|
| SEI oracle predictions (test + synthetic, batch_size=8) | ~35 seconds |
| MSE + KS computation | < 1 second |
| JS distance (6-mer, 7497 seqs) | ~3 minutes |
| AUROC (CNN training, 50 epochs) | ~2 minutes |
| **Total (all 4 metrics)** | **~6 minutes** |

## Baseline results

Promoter checkpoint `model-epoch=175-val_loss=1119.9065.ckpt`, 7497 test-conditioned samples, 128 sampling steps:

```json
{
  "mse": 0.0335,
  "ks_statistic": 0.1654,
  "js_distance": 0.1085,
  "auroc": 0.9996
}
```

### Interpretation

- **MSE 0.034**: Moderate functional fidelity — generated sequences produce somewhat similar SEI predictions to real test sequences.
- **KS 0.165**: Some distributional shift in oracle predictions between real and generated.
- **JS 0.109**: K-mer composition is reasonably similar but not identical.
- **AUROC 0.9996**: A classifier can almost perfectly distinguish real from generated sequences, suggesting the model has not yet fully captured the fine-grained statistical structure of real promoter sequences. This is expected for epoch 175/300.

## Issues encountered and fixes

### 1. scipy not found on compute nodes

**Problem**: `ModuleNotFoundError: No module named 'scipy'` when running via `srun` with background execution.

**Fix**: scipy was already installed in the conda env. The issue was with conda activation in background srun jobs. Running the command in foreground mode resolved it. If this persists, explicitly activate:
```bash
srun bash -c 'eval "$(conda shell.bash hook 2>/dev/null)"; conda activate d3-dna; python evaluate.py ...'
```

### 2. SEI model architecture dependency

**Problem**: The SEI model (`d3_dna/sei.py`) is a complex architecture copied from the evaluation pipeline. It requires scipy for B-spline computations internally.

**Fix**: Ensured scipy is installed in the d3-dna environment (it was already present as a transitive dependency).

## Implementation notes

The evaluation script (`examples/promoter/evaluate.py`) implements all 4 metrics inline — no dependency on the external `d3_evaluation_pipeline` repo. The SEI model architecture is in `d3_dna/sei.py`.

Oracle predictions are computed once and shared between MSE and KS statistic tests. The JS distance and AUROC tests don't need the oracle and run independently.
