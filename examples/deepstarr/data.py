"""
DeepSTARR Dataset Loader

Loads Drosophila enhancer activity data from an H5 file and returns PyTorch
Datasets compatible with d3_dna.D3Trainer.

H5 file structure (verified against Zenodo data_DeepSTARR.h5):
    X_{split}: (N, 4, 249) float32 — one-hot encoded DNA, already NCHW
    Y_{split}: (N, 2) float32       — dual-head global activity (dev, hk)
    X_dinucShuff_{split} also exists in the file (dinucleotide-shuffled
    negative controls); we don't use it here.

DeepSTARR labels are GLOBAL (one scalar per output head, two heads total) —
broadcast across all positions by the embedding layer. The 2D label tensor
shape (B, 2) routes through the same path as K562/HepG2 (B, 1); only the
projection dim differs (set via cfg.dataset.signal_dim).

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
    """Resolve the DeepSTARR H5. Downloads from Zenodo on first use."""
    return resolve_path(
        spec=override,
        cache_dir=_cache_dir(cfg),
        filename=cfg.data.data_file,
        record=cfg.data.zenodo_record,
    )


def get_oracle_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the DeepSTARR oracle checkpoint. Downloads from Zenodo on first use."""
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


class DeepSTARRDataset(Dataset):
    """DeepSTARR dataset. Each item is (sequence_indices, [dev, hk] activity)."""

    def __init__(self, h5_path: Union[str, Path], split: str = "train"):
        with h5py.File(str(h5_path), "r") as f:
            x = np.array(f[f"X_{split}"])        # (N, 4, 249) — already NCHW
            labels = np.array(f[f"Y_{split}"])   # (N, 2)

        # (N, 4, 249) -> argmax over channel dim -> (N, 249) token indices
        self.X = torch.from_numpy(x).argmax(dim=1)  # (N, 249)
        self.y = torch.tensor(labels, dtype=torch.float32)
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
