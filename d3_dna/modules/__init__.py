"""
D3-DNA user-facing high-level API.

Exports the four orchestrator classes plus the abstract training-time callback:

    * D3Trainer           -- high-level training orchestrator
    * D3LightningModule   -- PL LightningModule for D3
    * D3DataModule        -- PL LightningDataModule for D3
    * D3Sampler           -- standalone sampling API
    * D3Evaluator         -- standalone evaluation runner (dataset-agnostic)
    * BaseSPMSEValidationCallback -- abstract PL callback for training-time SP-MSE
"""

from d3_dna.modules.trainer import D3Trainer, D3LightningModule, D3DataModule
from d3_dna.modules.sampler import D3Sampler
from d3_dna.modules.evaluator import D3Evaluator
from d3_dna.modules.callbacks import BaseSPMSEValidationCallback

__all__ = [
    "D3Trainer",
    "D3LightningModule",
    "D3DataModule",
    "D3Sampler",
    "D3Evaluator",
    "BaseSPMSEValidationCallback",
]
