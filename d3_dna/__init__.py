"""
d3_dna -- DNA Discrete Diffusion (D3) package.

Public API:
    D3Trainer    -- train a D3 model on any DNA dataset
    D3Sampler    -- generate sequences from a trained checkpoint
    D3Evaluator  -- evaluate generated sequences (dataset-agnostic)

Lower-level layout:
    d3_dna.models       -- pure PyTorch: TransformerModel, ConvolutionalModel, EMA,
                           diffusion math (Noise, Graph, Predictor, samplers, score-fn)
    d3_dna.modules      -- user-facing PL + standalone orchestrators (above classes)
    d3_dna.evals        -- dataset-agnostic evaluation metrics
    d3_dna.utils        -- shared DNA helpers
"""

from d3_dna.modules import (
    D3Trainer,
    D3LightningModule,
    D3DataModule,
    D3Sampler,
    D3Evaluator,
    BaseSPMSEValidationCallback,
)

__all__ = [
    "D3Trainer",
    "D3Sampler",
    "D3Evaluator",
    "D3LightningModule",
    "D3DataModule",
    "BaseSPMSEValidationCallback",
]

__version__ = "0.1.0"
