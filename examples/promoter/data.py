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
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PromoterDataset(Dataset):

    def __init__(self, data_file, split='train'):
        data = np.load(data_file)
        raw = data[split]  # (N, 1024, 6)

        seq_onehot = raw[:, :, :4]            # (N, 1024, 4)
        self.X = torch.tensor(seq_onehot, dtype=torch.float32).argmax(dim=-1)  # (N, 1024)
        self.y = torch.tensor(raw[:, :, 4:5], dtype=torch.float32)             # (N, 1024, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
