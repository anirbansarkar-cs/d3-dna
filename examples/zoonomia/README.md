# Zoonomia example — basic dataloader

A dataloader over the 241-species Zoonomia multi-genome alignment, restricted (for now) to human-centered ENCODE cCRE windows. Mirrors the [`examples/minimal/`](../minimal/) scaffold but only the dataloader is implemented — training / sampling / evaluation are deliberately stubbed because the X/Y shapes diverge from D3's standard contract. See "Not wired for training" at the bottom.

## Data sources

All inputs live on the Koo-lab cluster (read-only, do **not** write):

- `/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5` — 67 GB. Top-level groups `chr1`..`chr22`, `chrX`, `chrY`. Each group has `seq` of shape `(241, chrom_length)` `uint8` and a `phyloP` track (unused). Species axis 0 = `Homo_sapiens`.
- `/grid/koo/home/shared/d3/data/zoonomia/GRCh38-cCREs.bed` — 2,348,854 cCRE regions. Tab-separated: `chrom  start  end  id1  id2  class`.

The data is not on Zenodo, so this example exposes **no** Zenodo path resolvers — `data.py` reads directly from the cluster paths.

## Nucleotide encoding

The H5 `seq` arrays carry raw `uint8` codes that double as one-hot channel indices:

| Code | Symbol | Meaning |
|---|---|---|
| 0 | N | gap / unaligned |
| 1 | A | adenine |
| 2 | C | cytosine |
| 3 | G | guanine |
| 4 | T | thymine |

`F.one_hot(seq.long(), num_classes=5)` is therefore a no-op remap.

## cCRE class index

`y[..., 0]` is a class index in `[0..7]` per `CRE_CLASSES` in `data.py` (alphabetical, fixed):

| Index | Class       | Notes                              |
|------:|:------------|:-----------------------------------|
| 0     | CA          | chromatin accessible               |
| 1     | CA-CTCF     | CA + CTCF binding                  |
| 2     | CA-H3K4me3  | CA + H3K4me3                       |
| 3     | CA-TF       | CA + transcription factor          |
| 4     | PLS         | promoter-like signature            |
| 5     | TF          | transcription factor               |
| 6     | dELS        | distal enhancer-like signature     |
| 7     | pELS        | proximal enhancer-like signature   |

## Shape contract

Per `__getitem__`:

- `X` — `(350, 5)` float32 one-hot. Channels are `[N, A, C, G, T]`.
- `y` — `(1, 2)` float32, the row `[[class_idx, species_idx]]`.

A `DataLoader(batch_size=B, ...)` collates to `X: (B, 350, 5)` and `y: (B, 1, 2)`.

## Design choices

- **Human-only (`species_index=0`)** for now. Multi-species enumeration is deferred — the lazy `__getitem__` makes it easy to add later (extend the index tuples with a species axis, optionally guarded by an alignment-presence mask).
- **Split by chromosome.** Defaults: train = chr1..chr18, val = chr19..chr21, test = chr22 + chrX. The H5 alignment does not include chrY. Override via the `chromosomes=` kwarg.
- **Center-expand on the cCRE midpoint** to reach a fixed 350 bp window: `mid = (start + end) // 2`, window = `[mid - 175, mid + 175)`. Most cCREs are shorter than 350 bp (mean ~267 bp), so this pulls in real flanking sequence rather than synthetic padding.
- **Edge padding with N.** When a window runs off a chromosome boundary, the missing side is padded with `0`s (the N channel).
- **Lazy, worker-safe H5.** The 67 GB file is far too large to load into RAM (unlike `examples/k562/data.py`, which slurps its H5 with `np.array`). `ZoonomiaDataset` opens the H5 on first `__getitem__` per process and drops the handle on pickling so `DataLoader(num_workers > 0)` opens one handle per worker.

## Usage

```python
from data import ZoonomiaDataset
from torch.utils.data import DataLoader

train_ds = ZoonomiaDataset(split="train")
val_ds   = ZoonomiaDataset(split="val")
test_ds  = ZoonomiaDataset(split="test")

X, y = train_ds[0]              # X: (350, 5) float32; y: (1, 2) float32
loader = DataLoader(train_ds, batch_size=64, num_workers=2, shuffle=True)
for X, y in loader:             # X: (64, 350, 5); y: (64, 1, 2)
    ...
```

Smoke test:

```bash
cd examples/zoonomia
python data.py
```

prints split sizes and asserts shapes / dtype / one-hot.

## Not wired for training

This example only ships the dataloader. The other files (`train.py`, `sample.py`, `evaluate.py`, `oracle.py`, `callbacks.py`, `config_*.yaml`) are stubs because the dataloader output deviates from D3's standard training contract:

- **`X` is `(B, 350, 5)` one-hot float**, whereas `D3LightningModule` and `d3_dna/models/transformer.py:EmbeddingLayer` expect `(B, L) LongTensor` token ids. Before wiring training, either collapse `X` to token ids via `argmax(-1)` (yielding tokens in `[0..4]` with `num_classes=5`) or add a one-hot-consuming embedding head.
- **`y` is `(B, 1, 2)`**, which does not match `(B, signal_dim)` (global) or `(B, L, signal_dim)` (per-position) — the two shapes `EmbeddingLayer.forward` dispatches on.
- **Class index stored as float.** `y[..., 0]` is a discrete index riding in a FloatTensor; downstream code will likely want a separate Long field once training is wired.

Until those are addressed, `train.py` / `sample.py` print a banner and exit non-zero rather than silently misroute.
