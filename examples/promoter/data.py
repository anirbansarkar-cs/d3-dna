"""
Promoter Dataset Loader

Loads promoter NPZ files for D3 sampling and training.

Data format:
    The NPZ file contains arrays with shape (N, 1024, 6):
      - Channels 0-3: one-hot encoded DNA sequence (A, C, G, T)
      - Channel 4: per-position regulatory activity signal (used as label)
      - Channel 5: second regulatory signal (unused)

    Unlike K562/LentiMPRA where labels are a single scalar per sequence (global
    conditioning), promoter labels are PER-POSITION: each of the 1024 base pair
    positions has its own regulatory activity value.

    Dataset returns:
      X: (N, 1024) int64 — token indices
      y: (N, 1024, 1) float32 — per-position labels

Path resolution (data + oracle weights):
    CLI override > local file in cache/ > download from Zenodo. Configs carry
    Zenodo record + filenames; nothing in this example assumes a specific
    cluster filesystem layout.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from d3_dna.utils import resolve_path

EXAMPLE_DIR = Path(__file__).resolve().parent


def _cache_dir(cfg) -> Path:
    cache = Path(cfg.data.cache_dir)
    return cache if cache.is_absolute() else EXAMPLE_DIR / cache


def get_data_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the promoter NPZ. Downloads from Zenodo on first use."""
    return resolve_path(
        spec=override,
        cache_dir=_cache_dir(cfg),
        filename=cfg.data.data_file,
        record=cfg.data.zenodo_record,
    )


def get_oracle_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the SEI oracle checkpoint. Downloads from Zenodo on first use."""
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


def get_sei_features_file(cfg, override: Optional[Union[str, Path]] = None) -> Path:
    """Resolve target.sei.names. Not on Zenodo — config default is the Koo-lab
    cluster path; pass --sei-features to override on other hosts."""
    spec = override if override is not None and str(override) else cfg.data.sei_features
    candidate = Path(spec)
    if not candidate.is_absolute():
        candidate = EXAMPLE_DIR / candidate
    if not candidate.exists():
        raise FileNotFoundError(
            f"sei_features file {candidate} not found. The default points at the "
            f"Koo-lab cluster path; on other hosts pass --sei-features /your/path/target.sei.names."
        )
    return candidate


class PromoterDataset(Dataset):
    def __init__(self, data_file: Union[str, Path], split: str = "train"):
        data = np.load(str(data_file))
        raw = data[split]  # (N, 1024, 6)

        seq_onehot = raw[:, :, :4]            # (N, 1024, 4)
        self.X = torch.tensor(seq_onehot, dtype=torch.float32).argmax(dim=-1)  # (N, 1024)
        self.y = torch.tensor(raw[:, :, 4:5], dtype=torch.float32)             # (N, 1024, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
