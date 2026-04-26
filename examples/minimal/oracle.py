"""
Template — oracle wrapper + loader for a new D3 example.

The oracle is a regression model that scores DNA sequences (e.g. SEI for
promoter, LegNet for K562/HepG2, the DeepSTARR CNN for DeepSTARR). It is
consumed in two places:

    callbacks.py:get_oracle_predictions  → in-training SP-MSE validation
    evaluate.py:_load_*_real / load(...) → end-of-run mse / ks metrics

Two pieces are required:

    class FooOracle:
        def predict(self, x: np.ndarray, batch_size: int, progress: bool) -> np.ndarray:
            # x is (N, 4, L) one-hot float; return (N, k) regression outputs

    def load(checkpoint_path, device, **kwargs) -> FooOracle:
        # Build the underlying model, load weights, return the wrapper.

Reference implementation: examples/k562/oracle.py (LegNet) — under ~250 lines
including the vendored model definition.
"""

from typing import Union
from pathlib import Path

import numpy as np
import torch


class MinimalOracle:
    """Skeleton oracle wrapper. Holds a trained model and exposes .predict()."""

    def __init__(self, model: torch.nn.Module, device: Union[str, torch.device]):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, x: np.ndarray, batch_size: int = 128, progress: bool = True) -> np.ndarray:
        """Score a batch of one-hot sequences.

        Args:
            x: (N, 4, sequence_length) float one-hot DNA.
            batch_size: per-forward minibatch.
            progress: emit a tqdm bar over batches.

        Returns:
            (N, signal_dim) numpy array — same shape as the dataset's labels.
        """
        raise NotImplementedError(
            "Iterate batches of x through self.model and return (N, k) np.ndarray. "
            "See examples/k562/oracle.py:LentiMPRAOracle.predict for a 15-line "
            "implementation."
        )


def load(checkpoint_path: Union[str, Path], device: Union[str, torch.device], **kwargs) -> MinimalOracle:
    """Build the oracle architecture, load weights, return the wrapper.

    The factory typically:
        1. Builds the bare nn.Module (architecture is dataset-specific).
        2. Loads the state_dict from checkpoint_path (handle both PL .ckpt and
           bare-dict formats; strip any 'model.' / 'module.' prefix).
        3. Moves the model to device and sets it to eval mode.
        4. Returns MinimalOracle(model, device).
    """
    raise NotImplementedError(
        "Build the oracle nn.Module, load checkpoint_path, return "
        "MinimalOracle(model.to(device).eval(), device). See "
        "examples/k562/oracle.py:load for the canonical pattern."
    )
