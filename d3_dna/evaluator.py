"""D3-DNA Evaluation.

Provides:
    D3Evaluator                  -- unified metrics runner (mse, ks, js, auroc)
    BaseSPMSEValidationCallback  -- training-time SP-MSE validation callback
"""

import os
import json
import torch
import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer

from d3_dna import sampling as sampling_module


# =============================================================================
# SP-MSE VALIDATION CALLBACK (training-time; unchanged public surface)
# =============================================================================

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
        if self.oracle_model is not None:
            self.oracle_model = self.oracle_model.to(device)

        val_sequences, val_targets = self._get_validation_data(trainer)
        val_targets = val_targets.to(device)
        seq_length = pl_module.config.dataset.sequence_length

        sampling_fn = sampling_module.get_pc_sampler(
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


# =============================================================================
# D3 EVALUATOR — unified metrics runner
# =============================================================================

class D3Evaluator:
    """Run the four D3 evaluation metrics on generated sequences.

    Usage::

        ev = D3Evaluator(dataset="promoter")
        results = ev.evaluate(
            samples="samples.npz",
            oracle_checkpoint="best.sei.model.pth.tar",
            tests=("mse", "ks", "js", "auroc"),
            kmer_ks=(6,),
            output_path="eval_results.json",
        )

    Supported dataset keys: 'promoter', 'deepstarr', 'lentimpra'. The promoter
    oracle follows the DDSM protocol: SEI predictions are masked to the 2350
    H3K4me3 tracks (target.sei.names) and averaged to a single scalar per
    sequence; evaluation defaults to the 40k FANTOM chr8/9 test split.
    DeepSTARR returns (N, 2) [dev, hk] activity; LentiMPRA returns (N, 1).
    """

    SUPPORTED = ("promoter", "deepstarr", "lentimpra")
    ALL_TESTS = ("mse", "ks", "js", "auroc")
    ORACLE_TESTS = {"mse", "ks"}

    DEFAULT_REAL_DATA = {
        "promoter": "/grid/koo/home/shared/d3/data/promoter/Promoter_data_40k.npz",
        "deepstarr": None,
        "lentimpra": None,
    }

    def __init__(self, dataset: str, device: Optional[str] = None):
        if dataset not in self.SUPPORTED:
            raise ValueError(f"Unsupported dataset {dataset!r}; pick one of {self.SUPPORTED}")
        self.dataset = dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- I/O helpers ----

    @staticmethod
    def _normalize_onehot(x) -> np.ndarray:
        """Accept (N,4,L), (N,L,4), path to .npz, or torch tensor; return (N,4,L) float32."""
        if isinstance(x, str):
            data = np.load(x)
            key = "arr_0" if "arr_0" in data.files else data.files[0]
            x = data[key]
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D sequence array, got shape {x.shape}")
        if x.shape[1] == 4:
            arr = x
        elif x.shape[2] == 4:
            arr = x.transpose(0, 2, 1)
        else:
            raise ValueError(f"Neither dim 1 nor dim 2 has size 4: {x.shape}")
        return arr.astype(np.float32)

    def _default_real_data_path(self):
        path = self.DEFAULT_REAL_DATA[self.dataset]
        if path is None:
            raise NotImplementedError(
                f"No default real-data path for dataset {self.dataset!r}; pass real_data="
            )
        return path

    def _load_real_from_path(self, path):
        """Load raw test one-hot (N,4,L) from a dataset-specific path."""
        if self.dataset == "promoter":
            return np.load(path)["test"][:, :, :4].transpose(0, 2, 1).astype(np.float32)
        raise NotImplementedError(
            f"Real-data loader for {self.dataset!r} not implemented; pass a preloaded array."
        )

    # ---- main entry point ----

    def evaluate(
        self,
        samples,
        oracle_checkpoint: Optional[str] = None,
        real_data=None,
        tests=ALL_TESTS,
        kmer_ks=(6,),
        output_path: Optional[str] = None,
        paired_repeat: int = 1,
    ) -> dict:
        from d3_dna.evals.oracles import load_oracle
        from d3_dna.evals.functional import compute_fidelity_mse, compute_ks_statistic
        from d3_dna.evals.sequence import compute_js_spectrum, compute_auroc

        tests = tuple(tests)
        unknown = set(tests) - set(self.ALL_TESTS)
        if unknown:
            raise ValueError(f"Unknown tests: {sorted(unknown)}; valid: {self.ALL_TESTS}")

        if real_data is None:
            real_data = self._default_real_data_path()
        if isinstance(real_data, str):
            x_real = self._load_real_from_path(real_data)
        else:
            x_real = self._normalize_onehot(real_data)
        x_gen = self._normalize_onehot(samples)

        if paired_repeat > 1:
            x_real = np.repeat(x_real, paired_repeat, axis=0)
            print(f"[{self.dataset}] real tiled ×{paired_repeat} -> {x_real.shape}")

        if len(x_real) != len(x_gen):
            n = min(len(x_real), len(x_gen))
            print(f"[{self.dataset}] truncating to paired N={n} (real={len(x_real)}, gen={len(x_gen)})")
            x_real, x_gen = x_real[:n], x_gen[:n]
        print(f"[{self.dataset}] real={x_real.shape} gen={x_gen.shape}")

        results: dict = {}
        needs_oracle = any(t in self.ORACLE_TESTS for t in tests)
        pred_real = pred_gen = None
        if needs_oracle:
            if oracle_checkpoint is None:
                raise ValueError("oracle_checkpoint required for tests including mse/ks")
            oracle = load_oracle(self.dataset, oracle_checkpoint, self.device)
            print("[oracle] predicting on real...")
            pred_real = oracle.predict(x_real)
            print("[oracle] predicting on generated...")
            pred_gen = oracle.predict(x_gen)
            del oracle
            if self.device == "cuda":
                torch.cuda.empty_cache()

        for t in tests:
            print(f"=== {t} ===")
            if t == "mse":
                v = compute_fidelity_mse(pred_real, pred_gen)
                results["fidelity_mse"] = v
                print(f"  fidelity_mse: {v:.6f}")
            elif t == "ks":
                v = compute_ks_statistic(pred_real, pred_gen, progress=True)
                results["ks_statistic"] = v
                print(f"  ks_statistic: {v:.6f}")
            elif t == "js":
                spec = compute_js_spectrum(x_real, x_gen, kmer_ks)
                results["js_distance"] = {f"k{k}": v for k, v in spec.items()}
                for k, v in spec.items():
                    print(f"  js_distance k={k}: {v:.6f}")
            elif t == "auroc":
                v = compute_auroc(x_real, x_gen, device=self.device)
                results["auroc"] = v
                print(f"  auroc: {v:.6f}")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"saved -> {output_path}")

        return results
