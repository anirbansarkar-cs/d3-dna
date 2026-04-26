# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`d3-dna` is a pip-installable library extracted from the D3 (DNA Discrete Diffusion) research codebase. The core package is dataset-agnostic; each dataset (promoter, k562, deepstarr, hepg2, minimal) lives under `examples/` and provides its own Dataset class, oracle, SP-MSE callback, and `train.py`/`sample.py`/`evaluate.py` scripts that consume the public API.

Not yet on PyPI — install from source: `pip install -e .`. See `INSTALLATION.md` for cluster-specific pitfalls (torch pin `<2.6.0` for V100, `flash-attn==2.5.8` for GLIBC 2.28, build isolation/CUDA_HOME issues).

## Commands

Single-example loop (from `examples/<name>/`):

```bash
python train.py [--config config_transformer.yaml] [--work-dir outputs/...] [--resume path/to/last.ckpt]
python sample.py --checkpoint outputs/.../last.ckpt [--use-test-labels] [--steps 128] [--batch-size 64]
python evaluate.py --samples generated/samples.npz [--tests mse,ks,js,auroc]
```

SLURM wrappers exist alongside the scripts (`run_train.sh`, `run_sample_eval.sh`, `run_minimal.sh`) — they `cd "$(dirname "$0")"` then forward args, so `sbatch run_train.sh --config config_conv.yaml` works. Batch submissions must run on GPU nodes; H100 is required for flash attention (`--constraint="h100"`). V100s work only without flash-attn installed (falls back to SDPA automatically).

Quick import sanity check:

```bash
python -c "from d3_dna import D3Trainer, D3Sampler, D3Evaluator; from d3_dna.models import TransformerModel, ConvolutionalModel; print('OK')"
```

No `pytest` suite yet — `[dev]` extras declare it, but there are no tests checked in.

## Architecture

Public API is exactly four classes, re-exported from `d3_dna/__init__.py`:

| Class | Role |
|---|---|
| `D3Trainer` | Builds `pl.Trainer`, wires `D3LightningModule` + `D3DataModule`, runs `.fit(train_ds, val_ds, resume_from=...)`. Accepts config path or `OmegaConf` object, plus user callbacks. |
| `D3Sampler` | Loads a checkpoint + user-supplied model instance; `.generate(...)` or `.generate_batched(...)` returns one-hot sequences; `.save()` writes NPZ+FASTA. |
| `D3Evaluator` | Dataset-agnostic dispatcher over `d3_dna/evals/metrics.py`. Caller passes pre-loaded samples, real data, and an oracle with a `.predict(x)` method. Tests: `mse`, `ks` (oracle-based), `js`, `auroc`. |
| `BaseSPMSEValidationCallback` | Abstract PL callback for in-training SP-MSE validation. One concrete subclass per dataset lives in `examples/<name>/callbacks.py`. |

Package layout:

```
d3_dna/
├── models/      # TransformerModel (DDiT), ConvolutionalModel, EMA, diffusion math
│                # (Noise, Graph, Predictor, get_pc_sampler, get_score_fn)
├── modules/     # trainer.py, sampler.py, evaluator.py, callbacks.py, checkpoint.py
├── evals/       # metric implementations (MSE, KS, JS, AUROC)
└── utils/       # DNA helpers (sequences_to_strings, etc.)
```

### Key design invariants

- **Config-driven models.** `D3LightningModule.create_model()` dispatches on `cfg.model.architecture` (`transformer` or `convolutional`). Custom architectures require subclassing + overriding `create_model()` — do not add new `elif` branches.
- **Checkpoint formats.** `modules/checkpoint.py::load_checkpoint` handles both PL `.ckpt` (strips `score_model.` prefix) and the original D3 `{'model','ema','step'}` format. EMA weights are applied to the model before returning.
- **Config embedded in checkpoint.** Training configs are saved via `self.save_hyperparameters()` into `ckpt['hyper_parameters']['cfg']`. Extract with `OmegaConf.create(dict(ckpt['hyper_parameters']['cfg']))` when you've lost the YAML.
- **Global vs per-position conditioning is shape-determined, not flag-determined.** `EmbeddingLayer.forward` in `d3_dna/models/transformer.py` inspects `signal_embed.dim()`: 2D label `(B, signal_dim)` → unsqueezed and broadcast (K562, LentiMPRA, DeepSTARR); 3D label `(B, seq_len, signal_dim)` → added element-wise (Promoter). `dataset.signal_dim` only sizes the linear projection; it does NOT switch modes.
- **Dataset-specific logic stays in `examples/`.** Real-data layout, oracle construction, strand averaging, masking, DDSM tiling — none of this belongs in `d3_dna/`. `D3Evaluator` deliberately takes pre-loaded real data and a pre-loaded oracle.
- **`examples/promoter/legacy/`** is a separate sampling-only path for pretrained hybrid SEDD checkpoints; it is isolated from the from-scratch training flow and should not be merged with it.

### Config conventions

Each example has its own `config*.yaml`. Required top-level keys: `dataset` (name, sequence_length, num_classes, signal_dim), `ngpus`, `tokens`, `model`, `training`, `optim`, `graph`, `noise`, `sampling`, `eval`, and usually `wandb` + `paths`. `dataset.sequence_length` and `dataset.signal_dim` feed the model constructors; the actual data loading is a user-provided `Dataset` (see `examples/*/data.py`).

## Shared/cluster paths

The Koo-lab cluster has shared data + oracle weights at `/grid/koo/home/shared/d3/`:

- `data/promoter/Promoter_data.npz` — `(N, 1024, 6)` NPZ with train/test/valid keys. Channels 0–3 = one-hot DNA, channel 4 = per-position CAGE signal (used as label), channel 5 unused.
- `oracle_weights/promoter/best.sei.model.pth.tar` — SEI oracle. Requires 4096 bp input, so sequences are symmetrically padded with uniform 0.25 background (see `SAMPLING.md`).
- `trained_weights/promoter_09242025/...` — reference checkpoints.

**Never write to `/grid/koo/home/shared/`.** Default generated outputs to `~/scratch/` or in-repo `outputs/` / `generated/` (both git-ignored).

## Performance notes (reference values)

Promoter, 1024 bp, 128 sampling steps, H100 NVL: constant **~5.9 seq/s** regardless of batch size (compute-bound). ~27 MB/sample memory. Recommended batch size 512 — flash-attn 2.5.8 becomes unstable above ~2048. Full 7497-sample test set ≈ 21 minutes wall clock.

Full eval (all 4 metrics on 7497 samples) ≈ 6 minutes on H100 NVL. JS distance and AUROC dominate; MSE+KS share a single oracle forward pass.
