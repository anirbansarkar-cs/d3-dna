"""
D3-DNA Evaluation

Provides BaseEvaluator, BaseSPMSEValidationCallback, and the concrete D3Evaluator.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer

from d3_dna import sampling as sampling_module
from d3_dna.io import get_score_fn


# =============================================================================
# SP-MSE VALIDATION CALLBACK
# =============================================================================

class BaseSPMSEValidationCallback(Callback, ABC):
    """
    Base callback for evaluating generated sequences using oracle models and computing SP-MSE.

    Subclass this and implement the abstract methods for your dataset's oracle.
    """

    def __init__(
        self,
        oracle_path: str,
        data_path: str,
        validation_freq_epochs: int = 4,
        validation_samples: int = 1000,
        enabled: bool = True,
        sampling_steps: Optional[int] = None,
        early_stopping_patience: Optional[int] = None
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
        self.best_sp_mse = float('inf')
        self.steps_since_improvement = 0
        self.validation_data_cache = None
        self.best_checkpoint_path = None

        if not self.enabled:
            return

        if self.sampling_steps is None:
            self.sampling_steps = self.get_default_sampling_steps()

    @abstractmethod
    def get_default_sampling_steps(self) -> int:
        pass

    @abstractmethod
    def load_oracle_model(self):
        pass

    @abstractmethod
    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        pass

    @abstractmethod
    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if not self.enabled:
            return
        if stage == 'fit':
            self.oracle_model = self.load_oracle_model()
            if self.oracle_model is not None:
                self.oracle_model = self.oracle_model.to(pl_module.device)
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
        all_sequences = []
        all_targets = []

        for batch in val_dataloader:
            sequences, targets = self.process_batch(batch)
            all_sequences.append(sequences)
            all_targets.append(targets)

        all_sequences = torch.cat(all_sequences, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if self.validation_samples > 0 and len(all_sequences) > self.validation_samples:
            indices = torch.randperm(len(all_sequences))[:self.validation_samples]
            all_sequences = all_sequences[indices]
            all_targets = all_targets[indices]

        self.validation_data_cache = (all_sequences, all_targets)
        return self.validation_data_cache

    def _run_sp_mse_validation(self, trainer: Trainer, pl_module: LightningModule):
        device = pl_module.device

        if self.oracle_model is not None:
            self.oracle_model = self.oracle_model.to(device)

        val_sequences, val_targets = self._get_validation_data(trainer)
        val_targets = val_targets.to(device)

        seq_length = pl_module.config.dataset.sequence_length

        sampling_fn = sampling_module.get_pc_sampler(
            pl_module.graph, pl_module.noise,
            (len(val_sequences), seq_length),
            'analytic', self.sampling_steps, device=device
        )

        with torch.no_grad():
            was_training = pl_module.training
            pl_module.eval()

            if hasattr(pl_module, 'ema') and pl_module.ema is not None:
                pl_module.ema.store(pl_module.score_model.parameters())
                pl_module.ema.copy_to(pl_module.score_model.parameters())

            try:
                generated_sequences = sampling_fn(pl_module.score_model, val_targets)
                val_score = self.get_oracle_predictions(val_sequences, device)
                generated_score = self.get_oracle_predictions(generated_sequences, device)
                sp_mse = (val_score - generated_score) ** 2
                mean_sp_mse = torch.mean(sp_mse).cpu().item()

                pl_module.log('sp_mse/validation', mean_sp_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                pl_module.log('sp_mse/best', self.best_sp_mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

                if mean_sp_mse < self.best_sp_mse:
                    self.best_sp_mse = mean_sp_mse
                    self.steps_since_improvement = 0

                    if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                        os.remove(self.best_checkpoint_path)

                    if hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir:
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
                if hasattr(pl_module, 'ema') and pl_module.ema is not None:
                    pl_module.ema.restore(pl_module.score_model.parameters())
                if was_training:
                    pl_module.train()


# =============================================================================
# BASE EVALUATOR
# =============================================================================

class BaseEvaluator:
    """
    Base evaluation framework for D3 models.

    Subclass this to add dataset-specific oracle loading and metrics.
    """

    def __init__(self, dataset_name: str = 'generic'):
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, checkpoint_path: str, config, architecture: str = 'transformer'):
        """Load model. Override for custom loading logic."""
        raise NotImplementedError("Subclasses must implement load_model()")

    def create_dataloader(self, config, split: str = 'test', batch_size: Optional[int] = None):
        """Create dataloader. Override for your dataset."""
        raise NotImplementedError("Subclasses must implement create_dataloader()")

    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """Load oracle model. Override for your dataset."""
        return None

    def get_sequence_length(self, config) -> int:
        """Get sequence length from config."""
        return config.dataset.sequence_length

    def sample_sequences_for_evaluation(self, checkpoint_path, config, dataloader, num_steps, architecture='transformer', show_progress=False, viz_logger=None, oracle_model=None, data_path=None):
        """Sample sequences for evaluation using PC sampler."""
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        model.eval()

        sequence_length = self.get_sequence_length(config)

        all_sampled = []
        all_targets = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                inputs = batch
                targets = None

            batch_size = inputs.shape[0]

            sampling_fn = sampling_module.get_pc_sampler(
                graph, noise, (batch_size, sequence_length),
                'analytic', num_steps, device=self.device,
                viz_logger=viz_logger
            )

            if targets is not None:
                targets = targets.to(self.device)

            sampled = sampling_fn(model, targets)
            all_sampled.append(sampled.cpu())
            if targets is not None:
                all_targets.append(targets.cpu())

        sampled_sequences = torch.cat(all_sampled, dim=0)
        target_labels = torch.cat(all_targets, dim=0) if all_targets else None

        return sampled_sequences, target_labels

    def compute_sp_mse(self, sampled_sequences, oracle_model, original_data):
        """Compute SP-MSE between sampled and original sequences.

        Uses oracle_model.predict() for batched inference (matching the
        original D3 evaluation), falling back to direct forward call.

        Args:
            sampled_sequences: Integer token tensor (N, seq_len) or one-hot (N, 4, seq_len).
            oracle_model: Oracle model with .predict() or callable forward.
            original_data: One-hot test data (N, 4, seq_len).
        """
        if oracle_model is None:
            return float('nan')

        with torch.no_grad():
            # Convert integer tokens to one-hot (N, 4, seq_len) if needed
            if sampled_sequences.dim() == 2 and sampled_sequences.shape[-1] != 4:
                sampled_onehot = F.one_hot(sampled_sequences.long(), num_classes=4).float()
                sampled_onehot = sampled_onehot.permute(0, 2, 1)
            elif sampled_sequences.dim() == 3 and sampled_sequences.shape[-1] == 4:
                sampled_onehot = sampled_sequences.permute(0, 2, 1).float()
            else:
                sampled_onehot = sampled_sequences.float()

            original_input = original_data.float()

            # Use oracle's prediction method (original D3 uses predict_custom)
            if hasattr(oracle_model, 'predict_custom'):
                sampled_preds = oracle_model.predict_custom(sampled_onehot)
                original_preds = oracle_model.predict_custom(original_input)
            elif hasattr(oracle_model, 'predict'):
                sampled_preds = oracle_model.predict(sampled_onehot)
                original_preds = oracle_model.predict(original_input)
            else:
                sampled_onehot = sampled_onehot.to(self.device)
                original_input = original_input.to(self.device)
                sampled_preds = oracle_model(sampled_onehot)
                original_preds = oracle_model(original_input)

            # Ensure both are tensors on the same device
            if not isinstance(sampled_preds, torch.Tensor):
                sampled_preds = torch.tensor(sampled_preds)
            if not isinstance(original_preds, torch.Tensor):
                original_preds = torch.tensor(original_preds)

            sp_mse = torch.mean((original_preds - sampled_preds) ** 2).item()

        return sp_mse

    def save_results(self, metrics: Dict, output_path: str):
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def print_results(self, metrics: Dict):
        """Print evaluation results."""
        print(f"\n{self.dataset_name} Evaluation Results:")
        print("=" * 40)
        for key, value in metrics.items():
            print(f"{key}: {value}")


# =============================================================================
# D3 EVALUATOR — Concrete implementation with config-driven model loading
# =============================================================================

class D3Evaluator(BaseEvaluator):
    """
    Concrete evaluator for D3 models.

    Uses io.load_checkpoint() for model loading. Subclass and override
    load_oracle_model() and compute_metrics() for dataset-specific evaluation.

    Example::

        class MyEvaluator(D3Evaluator):
            def load_oracle_model(self, path, data_path):
                return MyOracle.load(path)

            def compute_metrics(self, generated_seqs, oracle):
                preds = oracle(generated_seqs)
                return {'mean_activity': preds.mean().item()}
    """

    def __init__(self, dataset_name: str = 'generic'):
        super().__init__(dataset_name)

    def load_model(self, checkpoint_path, config, architecture='transformer'):
        from d3_dna.models import TransformerModel, ConvolutionalModel
        from d3_dna.io import load_checkpoint

        if architecture == 'transformer':
            model = TransformerModel(config)
        elif architecture == 'convolutional':
            model = ConvolutionalModel(config)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return load_checkpoint(checkpoint_path, model, config, self.device)

    def get_sequence_length(self, config):
        return config.dataset.sequence_length
