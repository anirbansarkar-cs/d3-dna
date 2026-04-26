# Minimal example — scaffold for a new dataset

A skeleton with every function signature `D3Trainer` / `D3Sampler` / `D3Evaluator` expects, and `NotImplementedError` bodies pointing at the parts you need to fill in. Copy this whole directory to `examples/<your-dataset>/` and walk top-to-bottom.

The simplest fully-populated reference is [`examples/k562/`](../k562/) — when in doubt, diff your file against k562's to see what a working version looks like. For a dataset with per-position labels, see [`examples/promoter/`](../promoter/) instead. For dual-head global labels, see [`examples/deepstarr/`](../deepstarr/).

## What you have to fill in

| File | What to implement | Reference |
|---|---|---|
| `config_transformer.yaml` / `config_conv.yaml` | `dataset.{sequence_length, signal_dim}`, `model.length`, the entire `data:` block (Zenodo record + filenames), `wandb.{project, name}` | `examples/k562/config_*.yaml` |
| `data.py` | `MinimalDataset.__init__` (read your file → set `self.X`, `self.y`) | `examples/k562/data.py:K562Dataset` |
| `oracle.py` | `MinimalOracle.predict` and `load(...)` | `examples/k562/oracle.py` |
| `callbacks.py` | `MinimalSPMSECallback.{get_default_sampling_steps, get_oracle_predictions}` | `examples/k562/callbacks.py` |
| `evaluate.py` | `_load_real(data_path)` (read test split → `(N, 4, L)` float32) | `examples/k562/evaluate.py:_load_k562_real` |
| `train.py`, `sample.py` | Nothing — the templates already use `MinimalDataset` / `MinimalSPMSECallback` and resolve all paths via `data.py`. Rename the imports if you renamed the classes. |

## What you do NOT have to fill in

The training loop, sampler, evaluator metric implementations (`mse`, `ks`, `js`, `auroc`), per-architecture precision policy, EMA handling, distributed orchestration, and Zenodo download caching are all in `d3_dna/` and shared across every example. You only own the dataset-specific seam.

## Per-position vs global labels

`y` shape decides the conditioning mode (no flag controls it):

- `(N, signal_dim)` — global per-sample label, broadcast across all positions. K562 / HepG2 (`signal_dim=1`), DeepSTARR (`signal_dim=2`).
- `(N, sequence_length, signal_dim)` — per-position label, added element-wise. Promoter (`signal_dim=1`).

`d3_dna.models.transformer.EmbeddingLayer.forward` inspects `signal_embed.dim()` and dispatches.

## Usage (once filled in)

```bash
python train.py    --config config_transformer.yaml
python sample.py   --config config_transformer.yaml
python evaluate.py --config config_transformer.yaml --samples-dir generated
# (and the same with --config config_conv.yaml)
```

## Floating-point precision

Identical to the other examples — see [`examples/promoter/README.md`](../promoter/README.md#floating-point-precision) for the full explanation. The package default is `bf16-mixed` for transformers and `16-mixed` (fp16 + GradScaler) for convolutional; only override `cfg.training.precision` if your checkpoint was trained under a different policy. Promoter is the one such exception currently in the repo.
