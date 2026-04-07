"""
K562 LentIMPRA Dataset Loader

Loads MPRA regulatory activity data from an H5 file and returns PyTorch Datasets
compatible with d3_dna.D3Trainer.

H5 file structure:
    onehot_{split}: (N, 230, 4) — one-hot encoded DNA sequences
    y_{split}:      (N, 1)      — regulatory activity measurements
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class K562Dataset(Dataset):
    """K562 MPRA dataset. Each item is (sequence_indices, activity_label)."""

    def __init__(self, h5_path: str, split: str = "train"):
        """
        Args:
            h5_path: Path to lenti_MPRA_K562_data.h5
            split: One of 'train', 'valid', 'test'
        """
        with h5py.File(h5_path, "r") as f:
            onehot = np.array(f[f"onehot_{split}"])  # (N, 230, 4)
            labels = np.array(f[f"y_{split}"])        # (N, 1) or (N,)

        # Convert one-hot to token indices: (N, 230, 4) -> (N, 4, 230) -> argmax -> (N, 230)
        self.X = torch.from_numpy(onehot).permute(0, 2, 1).argmax(dim=1)  # (N, 230)
        self.y = torch.tensor(labels, dtype=torch.float32)
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(1)  # (N,) -> (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_datasets(h5_path: str):
    """Load train, validation, and test splits."""
    return (
        K562Dataset(h5_path, "train"),
        K562Dataset(h5_path, "valid"),
        K562Dataset(h5_path, "test"),
    )
