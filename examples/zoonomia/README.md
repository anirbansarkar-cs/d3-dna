# Zoonomia example

A D3 example trained on human-centered ENCODE cCRE windows drawn from the 241-species Zoonomia multi-genome alignment. Unconditional discrete diffusion over the 5-token DNA alphabet `{N, A, C, G, T}`. Mirrors the [`examples/minimal/`](../minimal/) scaffold; `oracle.py`, `callbacks.py`, `evaluate.py`, `sample.py` remain stubs (no oracle yet).

## Data sources

All inputs live on the Koo-lab cluster (read-only):

- `/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5` — 67 GB. Top-level groups `chr1`..`chr22`, `chrX`. Each group has `seq` of shape `(241, chrom_length)` `uint8` and a `phyloP` track (unused). Species axis 0 = `Homo_sapiens`. The H5 has no `chrY`.
- `/grid/koo/home/shared/d3/data/zoonomia/GRCh38-cCREs.bed` — 2,348,854 cCRE regions. Tab-separated: `chrom  start  end  id1  id2  class`.

Not on Zenodo, so this example exposes no Zenodo path resolvers — `data.py` reads directly from the cluster paths.

## Token alphabet

The H5 `seq` arrays carry raw `uint8` codes that double as token ids — no remap needed:

| Token | Symbol | Meaning |
|------:|:-------|:--------|
| 0     | N      | gap / unaligned |
| 1     | A      | adenine |
| 2     | C      | cytosine |
| 3     | G      | guanine |
| 4     | T      | thymine |

`NUM_TOKENS = 5` in `data.py`. For human-only (`species_index=0`) the N token is effectively unused (cCREs sit deep inside chromosomes, fully aligned); mask the N logit at sampling time if zero-N generation is required.

## Shape contract

`ZoonomiaDataset.__getitem__(idx)` returns a single `LongTensor` of shape `(sequence_length,)` (default 350) with values in `[0, 5)`. A `DataLoader(batch_size=B)` collates to `LongTensor[B, L]` — directly compatible with `D3LightningModule.training_step`, which routes non-tuple batches through the unconditional loss path (`process_batch` → `(batch, None)`).

The per-sample metadata (cCRE class, species) is intentionally not returned — configs are unconditional (`signal_dim: 0`). The mapping is still accessible via `ds._index[idx] → (chrom, midpoint, class_idx)` for callers that want it, and `CRE_CLASSES` documents the class-index ordering:

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

## Design choices

- **Human-only (`species_index=0`)** for now. Multi-species enumeration is deferred — the lazy `__getitem__` makes it easy to add later.
- **Split by chromosome.** Defaults: train = chr1..chr18, val = chr19..chr21, test = chr22 + chrX. Override via the `chromosomes=` kwarg.
- **Center-expand on the cCRE midpoint** to reach a fixed 350 bp window: `mid = (start + end) // 2`, window = `[mid - 175, mid + 175)`. Most cCREs are shorter than 350 bp (mean ~267 bp), so this pulls in real flanking sequence.
- **Edge padding with N.** When a window runs off a chromosome boundary, the missing side is padded with token id 0.
- **Lazy, worker-safe H5.** The 67 GB file is far too large to load into RAM. `ZoonomiaDataset` opens the H5 lazily and drops the handle on pickling, so `DataLoader(num_workers > 0)` opens one handle per worker. Under DDP, `train.py` switches multiprocessing to `spawn` (parent already initialized CUDA; forked workers + pin_memory abort).

## Usage

```python
from data import ZoonomiaDataset
from torch.utils.data import DataLoader

train_ds = ZoonomiaDataset(split="train")
batch = next(iter(DataLoader(train_ds, batch_size=64, num_workers=2, shuffle=True)))
# batch.shape == (64, 350); batch.dtype == torch.long; values in [0, 5)
```

Smoke test:

```bash
cd examples/zoonomia
python data.py        # prints split sizes; asserts shape (4, 350) long
```

## Training

```bash
python train.py --config config_transformer.yaml                # single GPU
python train.py --config config_transformer.yaml --ngpus 4      # DDP
python train.py --config config_conv.yaml --output-dir outputs/zoonomia_conv
python train.py --config config_transformer.yaml \
    --resume-from train_experiments/zoonomia_transformer/checkpoints/last.ckpt
```

The configs enable W&B at `d3-cshl / d3-dna-ccre` with a stable `wandb.id` so chained resume jobs fold into one run (`d3_dna.modules.trainer` passes `resume='allow'` whenever `cfg.wandb.id` is set). Run `wandb login` once before submitting.
