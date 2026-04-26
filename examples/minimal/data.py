"""
Template — Dataset class + Zenodo path resolvers for a new D3 example.

Three resolvers + one Dataset class is the entire surface that train.py /
sample.py / evaluate.py expect from this module. Reference implementation:
examples/k562/data.py (the simplest concrete example).

What the resolvers do:
    - Layered lookup: CLI override > local file in cache/ > download from Zenodo.
    - All three reuse d3_dna.utils.resolve_path; you only choose which filename
      and Zenodo record to pull, both read from cfg.data.

What the Dataset returns per __getitem__:
    - X: torch.LongTensor of shape (sequence_length,)
         token indices in [0, num_classes) — DNA bases as 0..3.
    - y: torch.FloatTensor
         shape (signal_dim,) for global per-sample labels (K562/HepG2/DeepSTARR),
         or (sequence_length, signal_dim) for per-position labels (Promoter).
         The shape determines whether EmbeddingLayer broadcasts or adds element-
         wise; you do not set a flag.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from d3_dna.utils import resolve_path

EXAMPLE_DIR = Path(__file__).resolve().parent


def _cache_dir(cfg) -> Path:
    cache = Path(cfg.data.cache_dir)
    return cache if cache.is_absolute() else EXAMPLE_DIR / cache


def get_data_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the dataset file. Downloads from Zenodo on first use."""
    return resolve_path(
        spec=override,
        cache_dir=_cache_dir(cfg),
        filename=cfg.data.data_file,
        record=cfg.data.zenodo_record,
    )


def get_oracle_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the oracle checkpoint. Downloads from Zenodo on first use."""
    return resolve_path(
        spec=override,
        cache_dir=_cache_dir(cfg),
        filename=cfg.data.oracle_model,
        record=cfg.data.zenodo_record,
    )


def get_checkpoint_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the trained model checkpoint. Downloads from Zenodo on first use."""
    return resolve_path(
        spec=override,
        cache_dir=_cache_dir(cfg),
        filename=cfg.data.checkpoint,
        record=cfg.data.zenodo_record,
    )


class MinimalDataset(Dataset):
    """Skeleton dataset. Replace the body with logic for your file format.

    Expected to be constructible as ``MinimalDataset(path, split="train"|"valid"|"test")``
    and to populate ``self.X`` (LongTensor of token indices) and ``self.y``
    (FloatTensor of labels). See examples/k562/data.py:K562Dataset for a
    concrete H5-based implementation.
    """

    def __init__(self, data_file: Union[str, Path], split: str = "train"):
        raise NotImplementedError(
            "Load your dataset here. Set self.X (LongTensor[N, L]) and "
            "self.y (FloatTensor[N, signal_dim] or [N, L, signal_dim])."
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
