# HepG2 LentiMPRA example

End-to-end D3 conditional diffusion on the HepG2 LentiMPRA dataset: **230 bp regulatory sequences with a single global regression target** (shape `(N, 1)`) — one activity measurement per sequence, broadcast across all positions by the embedding layer. Mirror of the [K562 example](../k562/README.md) for the HepG2 cell line; same sequence length, same conditioning shape, same LegNet-style oracle. Two architectures: a transformer (12-block DDiT, ~30M params) and a convolutional model (256 channels).

## Prerequisites

- `d3-dna` installed from source (the package isn't on PyPI yet).
- GPU with Ampere architecture or newer if `flash-attn` is installed (H100 recommended); otherwise any CUDA GPU — the transformer falls back to PyTorch SDPA automatically.
- `curl` on PATH (used to fetch defaults from Zenodo).

**Data + oracle weights + pretrained checkpoints auto-download from [Zenodo record 19774653](https://zenodo.org/records/19774653) on first run** and cache under `examples/hepg2/cache/` (gitignored). To use a pre-existing copy on a shared filesystem instead, pass `--data-file /path/to/data_MPRA_HepG2.h5`, `--oracle-file /path/to/Oracle_MPRA_HepG2.ckpt`, or `--checkpoint /path/to/D3_Tran_MPRA_HepG2.ckpt` to any of the scripts.

> **Note on the conv checkpoint.** `D3_Conv_MPRA_HepG2.pth` ships in the SEDD legacy format (top-level `model` / `ema` / `step`), not Lightning `.ckpt`. `d3_dna.modules.checkpoint.load_checkpoint` dispatches by file content, so this is transparent — but it's why the conv config's `data.checkpoint` ends in `.pth`.

## Files

| File | Description |
|---|---|
| `config_transformer.yaml` | Transformer config (default). |
| `config_conv.yaml` | Convolutional config. |
| `data.py` | `HepG2Dataset` — loads the H5 and yields `(X: LongTensor[230], y: FloatTensor[1])`. Plus Zenodo path resolvers. |
| `oracle.py` | Vendored LegNet oracle (no `mpralegnet` install needed). Identical architecture to K562's. |
| `callbacks.py` | `HepG2MSECallback` — periodic SP-MSE validation against LegNet during training. |
| `train.py` | Training script using `D3Trainer`. |
| `sample.py` | Conditional sampling using `D3Sampler` (N replicates). |
| `evaluate.py` | Evaluation via `D3Evaluator` (MSE, KS, JS, AUROC) + LegNet oracle. |

## Usage

### 1. Train

```bash
# Transformer
python train.py --config config_transformer.yaml

# Convolutional
python train.py --config config_conv.yaml --output-dir outputs/hepg2_conv

# Resume
python train.py --config config_transformer.yaml \
    --resume-from outputs/hepg2_transformer/checkpoints/last.ckpt
```

Checkpoints land in `outputs/hepg2_{architecture}/checkpoints/`. `HepG2MSECallback` tracks SP-MSE periodically during training using the LegNet oracle.

Submit on SLURM (replace `<env>` and the conda init):

```bash
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
    --job-name=d3_hepg2_train --wrap="source /path/to/conda.sh && conda activate <env> && \
        python train.py --config config_transformer.yaml"
```

### 2. Sample

HepG2 labels are a single global scalar; **the default behaviour is to condition on the test-set activity labels** (one sequence per test record, ×N replicates). Pass `--random-labels` to draw scalar labels from a Gaussian instead.

```bash
# Default: condition on test labels, 5 replicates × N_test sequences,
# 20 sampling steps, transformer Zenodo checkpoint auto-downloaded.
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

Output: `generated/sample_{i}.npz` (one-hot `(N, 230, 4)`) and `generated/sample_{i}.fasta` per replicate.

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

`evaluate.py` loads the LegNet oracle (for MSE/KS), reads the real-data H5 (`onehot_test`), and dispatches through `D3Evaluator`. Outputs land in `eval_results/`: per-replicate JSON, plus `summary.csv` and `summary.json` with the across-replicate mean.

## Floating-point precision

Identical to the K562 example — see [`examples/k562/README.md`](../k562/README.md#floating-point-precision) for the per-stage summary, and [`examples/promoter/README.md`](../promoter/README.md#floating-point-precision) for the detailed narrative.

## Metrics

| Metric | Description | Direction |
|---|---|---|
| `fidelity_mse` | Paired MSE of LegNet oracle predictions (real vs generated) | Lower is better |
| `ks_statistic` | Mean per-feature two-sample KS on oracle predictions | Lower is better |
| `js_distance` | Jensen-Shannon **distance** between k-mer distributions. Default: single k=6. | Lower is better |
| `auroc` | AUROC of a CNN discriminator (real=1, gen=0) | Closer to 0.5 is better |

## Reference results

Run end-to-end from the public Zenodo artifacts (`D3_Tran_MPRA_HepG2.ckpt`, `D3_Conv_MPRA_HepG2.pth`, `data_MPRA_HepG2.h5`, `Oracle_MPRA_HepG2.ckpt`). Sampling: one sequence per HepG2 test record (`--use-test-labels`, default), 5 replicates, mean reported. JS reported at single `k=6`. Lower is better on every metric except AUROC, which targets 0.5.

| Architecture | `fidelity_mse` ↓ | `ks_statistic` ↓ | `js_distance` (k=6) ↓ | `auroc` (→ 0.5) |
|---|---|---|---|---|
| Transformer (12-block DDiT, ~30 M) | 0.4445 | 0.04400 | 0.02495 | 0.6464 |
| Convolutional (256-ch) | 0.4683 | 0.04097 | 0.05286 | 0.7954 |

Reproduce with:

```bash
python sample.py   --config config_transformer.yaml
python evaluate.py --config config_transformer.yaml --samples-dir generated --kmer-ks 6
# (and the same with --config config_conv.yaml)
```
