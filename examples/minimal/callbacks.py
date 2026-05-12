"""
Template — SP-MSE validation callback for a new D3 example.

Subclass d3_dna.BaseSPMSEValidationCallback and implement four small methods.
The base class handles the periodic-sampling + MSE-vs-ground-truth loop; you
only tell it which oracle to use, how to convert generated tokens into oracle
inputs, and how to unpack a batch.

Reference implementation: examples/k562/callbacks.py — 18 lines.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from d3_dna import BaseSPMSEValidationCallback
from oracle import load as load_minimal_oracle


class MinimalSPMSECallback(BaseSPMSEValidationCallback):

    def get_default_sampling_steps(self) -> int:
        """Number of PC sampling steps used to generate validation samples.
        Smaller = faster validation; larger = closer to final-eval fidelity."""
        raise NotImplementedError("Pick a small integer (k562 uses 20, promoter uses 128).")

    def load_oracle_model(self):
        """Construct and return the oracle. Called once at trainer setup."""
        return load_minimal_oracle(self.oracle_path, device="cuda")

    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Score `sequences` (LongTensor[N, L] of token ids) with the oracle.

        Typical body:
            onehot = F.one_hot(sequences.long(), num_classes=4).float()      # (N, L, 4)
            onehot_nchw = onehot.permute(0, 2, 1).cpu().numpy()              # (N, 4, L)
            preds = self.oracle_model.predict(onehot_nchw, batch_size=128, progress=False)
            return torch.tensor(preds, dtype=torch.float32, device=device)
        """
        raise NotImplementedError("Convert tokens -> one-hot -> oracle.predict; return torch.Tensor.")

    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack a dataloader batch into (sequences, labels). For the typical
        Dataset returning (X, y), this is just `return batch`."""
        X, y = batch
        return X, y
