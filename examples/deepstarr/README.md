# DeepSTARR example

End-to-end D3 conditional diffusion on the DeepSTARR Drosophila enhancer dataset (de Almeida et al. 2022): **249 bp regulatory sequences with a dual-head global regression target** (developmental + housekeeping activity, shape `(N, 2)`). Two architectures: a transformer (12-block DDiT, ~30M params) and a convolutional model (256 channels).

## Prerequisites

- `d3-dna` installed from source (the package isn't on PyPI yet).
- GPU with Ampere architecture or newer if `flash-attn` is installed (H100 recommended); otherwise any CUDA GPU — the transformer falls back to PyTorch SDPA automatically.
- `curl` on PATH (used to fetch defaults from Zenodo).

**Data + oracle weights + pretrained checkpoints auto-download from [Zenodo record 19774653](https://zenodo.org/records/19774653) on first run** and cache under `examples/deepstarr/cache/` (gitignored). To use a pre-existing copy on a shared filesystem instead, pass `--data-file /path/to/data_DeepSTARR.h5`, `--oracle-file /path/to/Oracle_DeepSTARR.ckpt`, or `--checkpoint /path/to/D3_Tran_DeepSTARR.ckpt`.

> **Note on the conv checkpoint.** `D3_Conv_DeepSTARR.pth` ships in the SEDD legacy format (top-level `model` / `ema` / `step`), not Lightning `.ckpt`. `d3_dna.modules.checkpoint.load_checkpoint` dispatches by file content, so this is transparent — but it's why the conv config's `data.checkpoint` ends in `.pth`.

## Files

| File | Description |
|---|---|
| `config_transformer.yaml` | Transformer config (default). |
| `config_conv.yaml` | Convolutional config. |
| `data.py` | `DeepSTARRDataset` — loads the H5 and yields `(X: LongTensor[249], y: FloatTensor[2])`. Plus Zenodo path resolvers. |
| `oracle.py` | Vendored DeepSTARR CNN (de Almeida et al., 2022). |
| `callbacks.py` | `DeepSTARRSPMSECallback` — periodic SP-MSE validation against the DeepSTARR oracle. |
| `train.py` | Training script using `D3Trainer`. |
| `sample.py` | Conditional sampling using `D3Sampler` (N replicates). |
| `evaluate.py` | Evaluation via `D3Evaluator` (MSE, KS, JS, AUROC) + DeepSTARR oracle. |

## Usage

### 1. Train

```bash
# Transformer
python train.py --config config_transformer.yaml

# Convolutional
python train.py --config config_conv.yaml --output-dir outputs/deepstarr_conv

# Resume
python train.py --config config_transformer.yaml \
    --resume-from outputs/deepstarr_transformer/checkpoints/last.ckpt
```

Checkpoints land in `outputs/deepstarr_{architecture}/checkpoints/`. `DeepSTARRSPMSECallback` tracks SP-MSE periodically during training, with the mean reduced over both N and the 2 oracle output heads.

### 2. Sample

DeepSTARR labels are dual-head global scalars; **the default behaviour is to condition on the test-set [dev, hk] activity labels** (one sequence per test record, ×N replicates). Pass `--random-labels` to draw scalar labels from a Gaussian instead.

```bash
# Default: condition on test labels, 5 replicates × N_test sequences
python sample.py

# Conv backbone (Zenodo .pth checkpoint)
python sample.py --config config_conv.yaml

# Higher-fidelity sampling
python sample.py --steps 100

# Single replicate at offset 3 (for SLURM array jobs)
python sample.py --replicates 1 --rep-offset 3 --steps 100

# Random scalar labels
python sample.py --random-labels --num-samples 1000
```

Output: `generated/sample_{i}.npz` (one-hot `(N, 249, 4)`) and `generated/sample_{i}.fasta` per replicate.

### 3. Evaluate

```bash
# Full evaluation (MSE, KS, JS, AUROC) across all sample_*.npz replicates
python evaluate.py --samples-dir generated

python evaluate.py --samples-dir generated --config config_conv.yaml

# Subset of metrics
python evaluate.py --samples-dir generated --tests mse,ks

# JS averaged over k ∈ {1..7}
python evaluate.py --samples-dir generated --kmer-ks 1-7
```

`evaluate.py` loads the DeepSTARR oracle (for MSE/KS), reads the real-data H5 (`onehot_test`), and dispatches through `D3Evaluator`. Outputs land in `eval_results/`: per-replicate JSON, plus `summary.csv` and `summary.json` with the across-replicate mean.

## Floating-point precision

Identical to the other examples — see [`examples/promoter/README.md`](../promoter/README.md#floating-point-precision) for the full explanation. DeepSTARR-specific point: `oracle.py` explicitly casts its input to `torch.float32` before the forward pass, so the evaluation path stays fp32 regardless of upstream autocast context.

## Metrics

| Metric | Description | Direction |
|---|---|---|
| `fidelity_mse` | Paired MSE of DeepSTARR oracle predictions (real vs generated), reduced over both samples and the 2 output heads | Lower is better |
| `ks_statistic` | Mean per-feature two-sample KS on oracle predictions | Lower is better |
| `js_distance` | Jensen-Shannon **distance** between k-mer distributions. Default: single k=6. | Lower is better |
| `auroc` | AUROC of a CNN discriminator (real=1, gen=0) | Closer to 0.5 is better |

## Reference results

Run end-to-end from the public Zenodo artifacts (`D3_Tran_DeepSTARR.ckpt`, `D3_Conv_DeepSTARR.pth`, `data_DeepSTARR.h5`, `Oracle_DeepSTARR.ckpt`). Sampling: one sequence per DeepSTARR test record (`--use-test-labels`, default), 5 replicates, mean reported. JS reported at single `k=6`. Lower is better on every metric except AUROC, which targets 0.5.

| Architecture | `fidelity_mse` ↓ | `ks_statistic` ↓ | `js_distance` (k=6) ↓ | `auroc` (→ 0.5) |
|---|---|---|---|---|
| Transformer (12-block DDiT, ~30 M) | 1.0982 | 0.02335 | 0.03659 | 0.5978 |
| Convolutional (256-ch) | 1.1391 | 0.02595 | 0.03653 | 0.5798 |

Reproduce with:

```bash
python sample.py   --config config_transformer.yaml
python evaluate.py --config config_transformer.yaml --samples-dir generated --kmer-ks 6
# (and the same with --config config_conv.yaml)
```
