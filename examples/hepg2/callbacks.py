"""HepG2-specific SP-MSE validation callback using the LegNet oracle."""

import torch
import torch.nn.functional as F
from typing import Tuple

from d3_dna import BaseSPMSEValidationCallback
from oracle import load as load_legnet_oracle


class HepG2MSECallback(BaseSPMSEValidationCallback):

    def get_default_sampling_steps(self) -> int:
        return 20

    def load_oracle_model(self):
        return load_legnet_oracle(self.oracle_path, device="cuda")

    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        onehot = F.one_hot(sequences.long(), num_classes=4).float()  # (N, L, 4)
        onehot_nchw = onehot.permute(0, 2, 1).cpu().numpy()  # (N, 4, L)
        preds = self.oracle_model.predict(onehot_nchw, batch_size=256, progress=False)  # (N, 1)
        return torch.tensor(preds, dtype=torch.float32, device=device)

    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = batch
        return X, y
