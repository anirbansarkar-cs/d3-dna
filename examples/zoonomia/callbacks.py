"""
Stub — SP-MSE validation callback for the Zoonomia example.

Not wired: the dataloader's X/Y shapes diverge from D3's training contract
(see examples/zoonomia/README.md). Once that's resolved, fill this in by
analogy with examples/k562/callbacks.py.
"""

from typing import Tuple

import torch

from d3_dna import BaseSPMSEValidationCallback
from oracle import load as load_zoonomia_oracle


class ZoonomiaSPMSECallback(BaseSPMSEValidationCallback):
    def get_default_sampling_steps(self) -> int:
        raise NotImplementedError(
            "Zoonomia callback is not wired. See README 'Not wired for training'."
        )

    def load_oracle_model(self):
        return load_zoonomia_oracle(self.oracle_path, device="cuda")

    def get_oracle_predictions(
        self, sequences: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Zoonomia callback is not wired. See README 'Not wired for training'."
        )

    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = batch
        return X, y
