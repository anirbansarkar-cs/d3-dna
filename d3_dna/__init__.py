"""
d3_dna -- DNA Discrete Diffusion (D3) package.

Public API:
    D3Trainer    -- train a D3 model on any DNA dataset
    D3Sampler    -- generate sequences from a trained checkpoint
    D3Evaluator  -- evaluate generated sequences

Lower-level modules:
    d3_dna.models     -- TransformerModel, ConvolutionalModel, EMA
    d3_dna.diffusion  -- get_noise, get_graph, get_loss_fn
    d3_dna.sampling   -- pc_sampler, get_pc_sampler, predictors
    d3_dna.io         -- load_checkpoint, load_config, sequence utilities
"""

from d3_dna.trainer import D3Trainer, D3LightningModule, D3DataModule
from d3_dna.sampling import D3Sampler
from d3_dna.evaluator import D3Evaluator

__all__ = [
    "D3Trainer",
    "D3Sampler",
    "D3Evaluator",
    "D3LightningModule",
    "D3DataModule",
]

__version__ = "0.1.0"
