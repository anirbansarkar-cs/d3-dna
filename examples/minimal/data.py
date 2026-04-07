"""Synthetic DNA dataset for the minimal example."""
import torch
from torch.utils.data import TensorDataset


def make_synthetic_dataset(n_samples: int = 1000, seq_len: int = 64) -> TensorDataset:
    """
    Returns a TensorDataset of (sequences, labels) where:
      sequences: int64 tokens in [0, 3], shape (n_samples, seq_len)
      labels: float32 scalar per sequence, shape (n_samples, 1)
    """
    seqs = torch.randint(0, 4, (n_samples, seq_len), dtype=torch.long)
    labels = torch.rand(n_samples, 1)
    return TensorDataset(seqs, labels)
