"""
Zoonomia 241-species dataset loader.

Reads ENCODE cCRE windows from the 241-way multi-genome alignment at
``/grid/koo/home/shared/d3/data/zoonomia/`` and yields fixed-length one-hot
sequences with class+species labels. Only the basic dataloader is implemented
here; the training/sampling/evaluation files in this example are stubs because
the X/Y shapes deliberately diverge from D3's standard contract (see README).

Data sources:
    zoonomia_241.h5     — 67 GB. Top-level groups chr1..chr22, chrX, chrY,
                          each with `seq` of shape (241, chrom_length) uint8
                          and `phyloP` of shape (chrom_length,). Species axis 0
                          is Homo_sapiens. Nucleotide codes are direct uint8
                          values: 0=N (gap/unaligned), 1=A, 2=C, 3=G, 4=T.
    GRCh38-cCREs.bed    — 2.35M cCRE regions, tab-delimited:
                          chrom  start  end  id1  id2  class
                          Eight classes total (see CRE_CLASSES below).

Output contract per __getitem__:
    X: torch.FloatTensor of shape (sequence_length, NUC_CHANNELS) — one-hot.
       Channel order mirrors the H5 uint8 codes directly, so
       F.one_hot(seq.long(), num_classes=5) is a no-op remap (no lookup table).
    y: torch.FloatTensor of shape (1, 2) — [[class_idx, species_idx]].

Both shapes deviate from D3's training-pipeline expectations
(LongTensor[L] + FloatTensor[signal_dim]); the divergence is intentional.

Unlike other examples, the data lives only on the Koo-lab cluster — there is
no Zenodo record, so this module exposes no get_data_file / get_oracle_file /
get_checkpoint_file resolvers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_H5 = Path("/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5")
DEFAULT_BED = Path("/grid/koo/home/shared/d3/data/zoonomia/GRCh38-cCREs.bed")

NUC_CHANNELS = 5  # 0=N, 1=A, 2=C, 3=G, 4=T (matches raw uint8 codes)

CRE_CLASSES = (
    "CA",
    "CA-CTCF",
    "CA-H3K4me3",
    "CA-TF",
    "PLS",
    "TF",
    "dELS",
    "pELS",
)
CRE_CLASS_TO_IDX = {c: i for i, c in enumerate(CRE_CLASSES)}

DEFAULT_CHROM_SPLITS: dict[str, list[str]] = {
    "train": [f"chr{i}" for i in range(1, 19)],
    "val": ["chr19", "chr20", "chr21"],
    "test": ["chr22", "chrX"],
}


def _load_bed_index(
    bed_path: Union[str, Path], kept_chroms: set[str]
) -> list[tuple[str, int, int]]:
    """Parse the cCRE BED into (chrom, midpoint, class_idx) tuples.

    Filters to rows whose chrom is in ``kept_chroms``. Midpoint uses integer
    floor division — even-length CREs land mid-leaning-low, which is stable.
    Unknown class strings raise KeyError (the eight known classes are
    exhaustive per the data audit).
    """
    out: list[tuple[str, int, int]] = []
    with open(bed_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            chrom = row[0]
            if chrom not in kept_chroms:
                continue
            start = int(row[1])
            end = int(row[2])
            cls_idx = CRE_CLASS_TO_IDX[row[5]]
            out.append((chrom, (start + end) // 2, cls_idx))
    return out


class ZoonomiaDataset(Dataset):
    """Human-centered cCRE windows from the Zoonomia 241-way alignment.

    Each item is a 350 bp window (default) centered on a cCRE midpoint in
    hg38, drawn from a single species (default: human, axis 0 of the H5
    `seq` datasets). Windows that fall off a chromosome edge are N-padded.

    The H5 file (67 GB) is opened lazily inside ``__getitem__`` and dropped
    on pickling so DataLoader workers each open their own handle.
    """

    def __init__(
        self,
        h5_path: Union[str, Path] = DEFAULT_H5,
        bed_path: Union[str, Path] = DEFAULT_BED,
        split: str = "train",
        sequence_length: int = 350,
        chromosomes: Optional[dict[str, list[str]]] = None,
        species_index: int = 0,
    ):
        if sequence_length % 2 != 0:
            raise ValueError(
                f"sequence_length must be even (got {sequence_length})"
            )
        chromosomes = chromosomes or DEFAULT_CHROM_SPLITS
        if split not in chromosomes:
            raise KeyError(
                f"unknown split {split!r}; expected one of {list(chromosomes)}"
            )

        self._h5_path = str(h5_path)
        self.species_index = int(species_index)
        self.sequence_length = int(sequence_length)
        self.half = self.sequence_length // 2

        kept = set(chromosomes[split])

        # One-shot pass: read chrom lengths, then close so __init__ doesn't
        # leave a handle open across pickling.
        with h5py.File(self._h5_path, "r") as f:
            self._chrom_lengths: dict[str, int] = {
                chrom: int(f[chrom].attrs["length"]) for chrom in kept
            }

        self._index = _load_bed_index(bed_path, kept)
        self._h5: Optional[h5py.File] = None

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        chrom, mid, class_idx = self._index[idx]

        L = self._chrom_lengths[chrom]
        lo = mid - self.half
        hi = mid + self.half
        lo_clip = max(lo, 0)
        hi_clip = min(hi, L)

        core = self._h5[chrom]["seq"][self.species_index, lo_clip:hi_clip]

        if lo < 0 or hi > L:
            pieces = []
            if lo < 0:
                pieces.append(np.zeros(-lo, dtype=np.uint8))
            pieces.append(np.asarray(core, dtype=np.uint8))
            if hi > L:
                pieces.append(np.zeros(hi - L, dtype=np.uint8))
            raw = np.concatenate(pieces)
        else:
            raw = np.asarray(core, dtype=np.uint8)

        seq = torch.from_numpy(np.ascontiguousarray(raw)).long()
        X = F.one_hot(seq, num_classes=NUC_CHANNELS).float()
        y = torch.tensor(
            [[float(class_idx), float(self.species_index)]],
            dtype=torch.float32,
        )
        return X, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    for split in ("train", "val", "test"):
        ds = ZoonomiaDataset(split=split)
        print(f"{split}: {len(ds):,} samples")

    ds = ZoonomiaDataset(split="train")
    X, y = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    assert X.shape == (4, 350, NUC_CHANNELS), X.shape
    assert X.dtype == torch.float32, X.dtype
    assert y.shape == (4, 1, 2), y.shape
    assert y.dtype == torch.float32, y.dtype
    assert X.sum(-1).eq(1).all(), "X is not one-hot"
    print("OK", X.shape, y.shape)
