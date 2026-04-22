# Legacy promoter sampling

Self-contained backwards-compatibility path for loading **pretrained** promoter
checkpoints that were produced outside of `d3_dna`'s native training pipeline.
Sampling-only — there is no training entry point here. For new training,
use `../train.py`.

## What lives here

| File | Purpose |
|---|---|
| `hybrid_shim.py` | `HybridSEDD` module reproducing the colleague's dual-tower SEDD module tree so the raw state_dict and EMA `shadow_params` load cleanly. Forward runs only the transformer path. |
| `sample_hybrid.py` | Sample from a hybrid SEDD checkpoint using `HybridSEDD` + `d3_dna.D3Sampler`. |
| `sample_ddsm_protocol.py` | 5-per-TSS sampling on the 40k chr8/9 test split for either a D3 checkpoint or an upstream DDSM checkpoint. |
| `ddsm_ref/` | Vendored `ddsm` code (ScoreNet, Euler–Maruyama sampler, noise factory) from [jzhoulab/ddsm](https://github.com/jzhoulab/ddsm). Python only — the 3.6 GB pre-sampled noise schedule is **not** bundled; pass its path via `--ddsm-schedule`. |

## Sampling from the hybrid tran checkpoint

```bash
cd examples/promoter/legacy
python sample_hybrid.py \
    --checkpoint /grid/koo/home/shared/d3/trained_weights/promoter/tran/checkpoint_50.pth \
    --use-test-labels \
    --steps 128 \
    --batch-size 64 \
    --output-dir generated_hybrid
```

The script runs a state_dict parity check before sampling and will assert-fail
loudly if the shim doesn't match the checkpoint.

## DDSM protocol comparison

```bash
cd examples/promoter/legacy
# D3 side
python sample_ddsm_protocol.py --model d3 \
    --checkpoint /path/to/d3_model.ckpt \
    --config ../config_transformer.yaml \
    --steps 100 --batch-size 64 \
    --output-dir generated/d3_5perTSS

# DDSM side
python sample_ddsm_protocol.py --model ddsm \
    --checkpoint /path/to/ddsm_checkpoint.pth \
    --ddsm-schedule /grid/koo/home/duran/scratch/ddsm_artifacts/steps400.cat4.speed_balance.time4.0.samples100000.pth \
    --steps 100 --batch-size 64 \
    --output-dir generated/ddsm_5perTSS
```

`--ddsm-schedule` is required when `--model=ddsm`. Expect a shared-drive path;
the schedule is ~3.6 GB.

## Stability caveat

`hybrid_shim.py` imports internals of `d3_dna.models.transformer` and
`d3_dna.models.layers` (`DDiTBlock`, `DDitFinalLayer`, `EmbeddingLayer`,
`Dense`, `LabelEmbedder`, `Rotary`, `TimestepEmbedder`). If those modules are
renamed or their constructors change, this shim will break. Pin to the
matching `d3-dna` version or update the shim in lockstep.

`ddsm_ref/` is a frozen copy of upstream `jzhoulab/ddsm` used only for
inference-time sampling. Do not modify it.
