"""
K562 LentiMPRA Dataset Loader

Loads MPRA regulatory activity data from an H5 file and returns PyTorch Datasets
compatible with d3_dna.D3Trainer.

H5 file structure:
    onehot_{split}: (N, 230, 4) — one-hot encoded DNA sequences
    y_{split}:      (N, 1)      — single global regulatory activity measurement

Unlike Promoter (per-position labels of shape (N, 1024, 1)), K562 labels are
GLOBAL — a single scalar per 230 bp sequence, broadcast across all positions
by the embedding layer.

Path resolution (data + oracle weights + checkpoints):
    CLI override > local file in cache/ > download from Zenodo. Configs carry
    Zenodo record + filenames; nothing in this example assumes a specific
    cluster filesystem layout.
"""

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from d3_dna.utils import resolve_path

EXAMPLE_DIR = Path(__file__).resolve().parent


def _cache_dir(cfg) -> Path:
    cache = Path(cfg.data.cache_dir)
    return cache if cache.is_absolute() else EXAMPLE_DIR / cache


def get_data_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the K562 H5. Downloads from Zenodo on first use."""
    return resolve_path(
        spec=override,
        cache_dir=_cache_dir(cfg),
        filename=cfg.data.data_file,
        record=cfg.data.zenodo_record,
    )


def get_oracle_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the LegNet oracle checkpoint. Downloads from Zenodo on first use."""
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


class K562Dataset(Dataset):
    """K562 MPRA dataset. Each item is (sequence_indices, activity_label)."""

    def __init__(self, h5_path: Union[str, Path], split: str = "train"):
        with h5py.File(str(h5_path), "r") as f:
            onehot = np.array(f[f"onehot_{split}"])  # (N, 230, 4)
            labels = np.array(f[f"y_{split}"])        # (N, 1) or (N,)

        # one-hot (N, 230, 4) -> (N, 4, 230) -> argmax -> (N, 230) token indices
        self.X = torch.from_numpy(onehot).permute(0, 2, 1).argmax(dim=1)  # (N, 230)
        self.y = torch.tensor(labels, dtype=torch.float32)
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(1)  # (N,) -> (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
