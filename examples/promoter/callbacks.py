"""Promoter-specific SP-MSE validation callback using the SEI oracle."""

import torch
import torch.nn.functional as F
from typing import Tuple

from d3_dna import BaseSPMSEValidationCallback
from oracle import load as load_sei_oracle


class PromoterSPMSECallback(BaseSPMSEValidationCallback):

    def __init__(self, *args, sei_features_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.sei_features_path = sei_features_path

    def get_default_sampling_steps(self) -> int:
        return 128

    def load_oracle_model(self):
        return load_sei_oracle(self.oracle_path, device="cuda",
                               target_names_path=self.sei_features_path)

    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        onehot = F.one_hot(sequences.long(), num_classes=4).float()  # (N, L, 4)
        onehot_nchw = onehot.permute(0, 2, 1).cpu().numpy()  # (N, 4, L)
        preds = self.oracle_model.predict(onehot_nchw, batch_size=64, progress=False)  # (N, 1)
        return torch.tensor(preds, dtype=torch.float32, device=device)

    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = batch
        return X, y
