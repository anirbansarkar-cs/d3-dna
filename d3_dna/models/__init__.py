"""
D3-DNA model architectures.

Provides TransformerModel and ConvolutionalModel, both fully config-driven
and dataset-agnostic.
"""

from d3_dna.models.ema import ExponentialMovingAverage
from d3_dna.models.transformer import TransformerModel
from d3_dna.models.cnn import ConvolutionalModel

__all__ = [
    "TransformerModel",
    "ConvolutionalModel",
    "ExponentialMovingAverage",
]
