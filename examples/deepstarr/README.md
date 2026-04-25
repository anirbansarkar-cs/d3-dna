# DeepSTARR example (placeholder)

**Status: oracle-only stub.** This directory currently contains just `oracle.py` (the DeepSTARR CNN, vendored from de Almeida et al. 2022). The `data.py`, `config.yaml`, `train.py`, `sample.py`, and `evaluate.py` that a complete example needs have not been written yet.

## What exists

| File | Description |
|---|---|
| `oracle.py` | DeepSTARR CNN — takes `(N, 4, 249)` one-hot, returns `(N, 2)` developmental/housekeeping activity predictions. |

## Building it out

Use `examples/k562/` as a template — the conditioning pattern is the same (global scalar labels, here two-dimensional). Steps:

1. Write a `data.py` with a `DeepSTARRDataset` that yields `(X: LongTensor[249], y: FloatTensor[2])`.
2. Copy `examples/k562/config.yaml`, set `dataset.sequence_length: 249`, `dataset.signal_dim: 2`.
3. Copy `train.py`, `sample.py`, `evaluate.py` from k562 and update the dataset import + any oracle-specific logic in `evaluate.py`.

## Floating-point precision

Identical to the other examples once the pipeline is built out — see [`examples/promoter/README.md`](../promoter/README.md#floating-point-precision) for the full explanation. DeepSTARR-specific point: `oracle.py` explicitly casts its input to `torch.float32` before the forward pass, so the evaluation path stays fp32 regardless of upstream autocast context.

## Reference results

Mean across replicates from a separate evaluation pipeline (not yet reproducible from this directory — see "Building it out"). JS reported at single `k=6`. Lower is better on every metric except AUROC, which targets 0.5.

| Architecture | `fidelity_mse` ↓ | `ks_statistic` ↓ | `js_distance` (k=6) ↓ | `auroc` (→ 0.5) |
|---|---|---|---|---|
| Convolutional | 1.1391 | 0.02595 | 0.03653 | 0.5798 |
| Transformer | 1.0982 | 0.02335 | 0.03659 | 0.5978 |
