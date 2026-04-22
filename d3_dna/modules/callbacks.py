"""
Training-time PL callbacks.

Only ``BaseSPMSEValidationCallback`` lives here — an abstract callback that
periodically samples from the model during training and scores the samples with
a dataset-specific oracle. Concrete subclasses (one per dataset) live in the
corresponding ``examples/<dataset>/`` directory.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from d3_dna.models.diffusion import get_pc_sampler


class BaseSPMSEValidationCallback(Callback, ABC):
    """Callback for periodic SP-MSE validation during training."""

    def __init__(
        self,
        oracle_path: str,
        data_path: str,
        validation_freq_epochs: int = 4,
        validation_samples: int = 1000,
        enabled: bool = True,
        sampling_steps: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
    ):
        super().__init__()
        self.oracle_path = oracle_path
        self.data_path = data_path
        self.validation_freq_epochs = validation_freq_epochs
        self.validation_samples = validation_samples
        self.enabled = enabled
        self.sampling_steps = sampling_steps
        self.early_stopping_patience = early_stopping_patience

        self.oracle_model = None
        self.best_sp_mse = float("inf")
        self.steps_since_improvement = 0
        self.validation_data_cache = None
        self.best_checkpoint_path = None

        if not self.enabled:
            return
        if self.sampling_steps is None:
            self.sampling_steps = self.get_default_sampling_steps()

    @abstractmethod
    def get_default_sampling_steps(self) -> int: ...

    @abstractmethod
    def load_oracle_model(self): ...

    @abstractmethod
    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor: ...

    @abstractmethod
    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if not self.enabled:
            return
        if stage == "fit":
            self.oracle_model = self.load_oracle_model()
            if self.oracle_model is not None:
                if hasattr(self.oracle_model, 'to'):
                    self.oracle_model = self.oracle_model.to(pl_module.device)
                if hasattr(self.oracle_model, 'eval'):
                    self.oracle_model.eval()
            else:
                self.enabled = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled or self.oracle_model is None:
            return
        current_epoch = trainer.current_epoch
        if current_epoch > 0 and current_epoch % self.validation_freq_epochs == 0:
            self._run_sp_mse_validation(trainer, pl_module)

    def _get_validation_data(self, trainer: Trainer):
        if self.validation_data_cache is not None:
            return self.validation_data_cache
        val_dataloader = trainer.datamodule.val_dataloader()
        all_sequences, all_targets = [], []
        for batch in val_dataloader:
            sequences, targets = self.process_batch(batch)
            all_sequences.append(sequences)
            all_targets.append(targets)
        all_sequences = torch.cat(all_sequences, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        if self.validation_samples > 0 and len(all_sequences) > self.validation_samples:
            idx = torch.randperm(len(all_sequences))[: self.validation_samples]
            all_sequences = all_sequences[idx]
            all_targets = all_targets[idx]
        self.validation_data_cache = (all_sequences, all_targets)
        return self.validation_data_cache

    def _run_sp_mse_validation(self, trainer: Trainer, pl_module: LightningModule):
        device = pl_module.device
        if self.oracle_model is not None and hasattr(self.oracle_model, 'to'):
            self.oracle_model = self.oracle_model.to(device)

        val_sequences, val_targets = self._get_validation_data(trainer)
        val_targets = val_targets.to(device)
        seq_length = pl_module.cfg.dataset.sequence_length

        sampling_fn = get_pc_sampler(
            pl_module.graph, pl_module.noise,
            (len(val_sequences), seq_length),
            "analytic", self.sampling_steps, device=device,
        )

        with torch.no_grad():
            was_training = pl_module.training
            pl_module.eval()
            if hasattr(pl_module, "ema") and pl_module.ema is not None:
                pl_module.ema.store(pl_module.score_model.parameters())
                pl_module.ema.copy_to(pl_module.score_model.parameters())
            try:
                generated_sequences = sampling_fn(pl_module.score_model, val_targets)
                if generated_sequences.unique().numel() <= 1:
                    print("[sp-mse] WARNING: sampling produced degenerate sequences, skipping validation")
                    return
                val_score = self.get_oracle_predictions(val_sequences, device)
                generated_score = self.get_oracle_predictions(generated_sequences, device)
                sp_mse = (val_score - generated_score) ** 2
                mean_sp_mse = torch.mean(sp_mse).cpu().item()

                pl_module.log("sp_mse/validation", mean_sp_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                pl_module.log("sp_mse/best", self.best_sp_mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

                if mean_sp_mse < self.best_sp_mse:
                    self.best_sp_mse = mean_sp_mse
                    self.steps_since_improvement = 0
                    if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                        os.remove(self.best_checkpoint_path)
                    if hasattr(trainer.logger, "save_dir") and trainer.logger.save_dir:
                        work_dir = trainer.logger.save_dir
                    else:
                        work_dir = trainer.default_root_dir
                    checkpoints_dir = os.path.join(work_dir, "checkpoints")
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    checkpoint_filename = f"sp-mse_{mean_sp_mse:.6f}_step_{trainer.global_step}.ckpt"
                    self.best_checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
                    trainer.save_checkpoint(self.best_checkpoint_path)
                else:
                    self.steps_since_improvement += 1
                    if (self.early_stopping_patience is not None and
                            self.steps_since_improvement >= self.early_stopping_patience):
                        trainer.should_stop = True
            finally:
                if hasattr(pl_module, "ema") and pl_module.ema is not None:
                    pl_module.ema.restore(pl_module.score_model.parameters())
                if was_training:
                    pl_module.train()
