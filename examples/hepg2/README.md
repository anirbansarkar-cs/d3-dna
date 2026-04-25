# HepG2 LentiMPRA example (placeholder)

Mirror of the [K562 example](../k562/README.md) for the HepG2 cell line (same 230 bp sequence length, same single-global-label conditioning, same LegNet-style oracle). See the K562 README for the canonical command reference.

## Known issue

The scripts in this directory (`train.py`, `sample.py`, `evaluate.py`) currently still `import K562Dataset` and reference `outputs/k562/…` paths — they're unmodified copies of the K562 scripts. Until that's fixed, **don't run this example as-is** — either use the K562 example, or wait for the hepg2 rewrite PR that swaps in `HepG2Dataset` and the hepg2 output/checkpoint directory.

## Files

| File | Description |
|---|---|
| `config.yaml` | HepG2-specific paths; otherwise identical to the K562 config. |
| `data.py` | Dataset class (intended to be HepG2-specific; currently aliased to K562). |
| `train.py`, `sample.py`, `evaluate.py` | Copied from K562 — **need rewriting** (see above). |
| `oracle.py` | Vendored LegNet oracle (same as K562). |

## Floating-point precision

Identical to the K562 example — see [`examples/k562/README.md`](../k562/README.md#floating-point-precision) for the per-stage summary, and [`examples/promoter/README.md`](../promoter/README.md#floating-point-precision) for the detailed narrative (fp16-mixed autocast on CUDA, LayerNorm fp32 island, score upcast to fp32 after `.exp()`, fp32 oracle forward).

## Reference results

Mean across replicates from a separate evaluation pipeline (not yet reproducible from this directory — see "Known issue"). JS reported at single `k=6`. Lower is better on every metric except AUROC, which targets 0.5.

| Architecture | `fidelity_mse` ↓ | `ks_statistic` ↓ | `js_distance` (k=6) ↓ | `auroc` (→ 0.5) |
|---|---|---|---|---|
| Convolutional | 0.4683 | 0.04097 | 0.05286 | 0.7954 |
| Transformer | 0.4445 | 0.04400 | 0.02495 | 0.6464 |
