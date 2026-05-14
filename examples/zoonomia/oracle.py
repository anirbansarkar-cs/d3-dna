"""
Stub — oracle wrapper for the Zoonomia example.

Not wired: this example currently ships only the dataloader. See README.md
("Not wired for training") for the X/Y shape divergences that block training,
sampling, and oracle-based evaluation. Once those are resolved, fill this in
by analogy with examples/k562/oracle.py (LegNet).
"""

from typing import Union
from pathlib import Path

import numpy as np
import torch


class ZoonomiaOracle:
    """Skeleton oracle wrapper. Holds a trained model and exposes .predict()."""

    def __init__(self, model: torch.nn.Module, device: Union[str, torch.device]):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(
        self, x: np.ndarray, batch_size: int = 128, progress: bool = True
    ) -> np.ndarray:
        raise NotImplementedError(
            "Zoonomia oracle is not wired. See examples/zoonomia/README.md "
            "'Not wired for training'."
        )


def load(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device],
    **kwargs,
) -> ZoonomiaOracle:
    raise NotImplementedError(
        "Zoonomia oracle is not wired. See examples/zoonomia/README.md "
        "'Not wired for training'."
    )
