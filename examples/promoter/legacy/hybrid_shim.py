"""
Eval-only shim for the colleague's hybrid SEDD checkpoint.

The checkpoint at /grid/koo/home/shared/d3/trained_weights/promoter/tran/
checkpoint_50.pth was produced by a single SEDD class that instantiated both a
transformer tower and a conv tower (shared embedders, two heads). Only the tran
tower was trained; the conv tower holds untouched init weights.

This class reproduces the colleague's module tree so:
  1. `checkpoint['model']` loads cleanly (all prefixes have a destination).
  2. `checkpoint['ema']['shadow_params']` zips 1:1 against
     `HybridSEDD.parameters()` so EMA copy-to targets the right tensors.

Forward runs only the transformer path — identical math to
`d3_dna.models.TransformerModel.forward`. The conv submodules sit in memory as
dead weights.

Use only for loading this specific checkpoint. For new training, use
`d3_dna.models.TransformerModel` directly.

Compat-only: depends on the internal layout of `d3_dna.models.transformer` and
`d3_dna.models.layers`. If those modules are renamed or their constructors
change, this shim will break and must be pinned to the matching d3-dna version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from d3_dna.models.transformer import DDiTBlock, DDitFinalLayer, EmbeddingLayer
from d3_dna.models.layers import Dense, LabelEmbedder, Rotary, TimestepEmbedder


class HybridSEDD(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)

        hidden = config.model.hidden_size
        n_heads = config.model.n_heads
        cond_dim = config.model.cond_dim
        n_blocks = config.model.n_blocks
        dropout = config.model.dropout
        class_dropout_prob = getattr(config.model, "class_dropout_prob", 0.1)

        num_classes = config.dataset.num_classes
        signal_dim = config.dataset.signal_dim
        n_channels = getattr(config.model, "conv_channels", 256)

        # --- Registration order mirrors the colleague's hybrid SEDD so EMA's
        # flat shadow_params list zips 1:1 against self.parameters(). ---

        self.scale = nn.Parameter(torch.ones(1))

        self.vocab_embed = EmbeddingLayer(hidden, vocab_size, signal_dim=signal_dim)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.label_embed = LabelEmbedder(num_classes, cond_dim, class_dropout_prob)
        self.rotary_emb = Rotary(hidden // n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(dim=hidden, n_heads=n_heads, cond_dim=cond_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.output_layer = DDitFinalLayer(hidden, vocab_size, cond_dim)

        # Conv tower: dead weights, kept only to match checkpoint layout.
        # Registration order (confirmed from state_dict insertion order):
        # linear -> conv_blocks -> denses -> norms -> final.
        self.linear = nn.Conv1d(vocab_size + 1, n_channels, kernel_size=9, padding=4)
        self.conv_blocks = self._create_conv_blocks(n_channels)
        # Dense(n_channels, n_channels): the colleague's Dense projects from
        # n_channels, not cond_dim (shape confirmed as (256, 256)).
        self.denses = nn.ModuleList([
            Dense(n_channels, n_channels) for _ in range(len(self.conv_blocks))
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, n_channels) for _ in range(len(self.conv_blocks))
        ])
        self.final = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n_channels, vocab_size, kernel_size=1),
        )

        self.scale_by_sigma = getattr(config.model, "scale_by_sigma", False)

    @staticmethod
    def _create_conv_blocks(n_channels: int) -> nn.ModuleList:
        dilation_pattern = [1, 1, 4, 16, 64]
        blocks = []
        for _ in range(4):
            for dilation in dilation_pattern:
                padding = 4 if dilation == 1 else 4 * dilation
                blocks.append(
                    nn.Conv1d(n_channels, n_channels, kernel_size=9,
                              dilation=dilation, padding=padding)
                )
        return nn.ModuleList(blocks)

    def forward(self, indices: torch.Tensor, labels: Optional[torch.Tensor] = None,
                train: bool = True, sigma: Optional[torch.Tensor] = None,
                layer_idx: Optional[int] = None):
        # Mirrors d3_dna.models.transformer.TransformerModel.forward (non-bridge path).
        x = self.vocab_embed(indices, labels)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        # Matches TransformerModel.forward: no inner autocast. The outer
        # get_score_fn autocast is the single precision-control point.
        rep = None
        for i, block in enumerate(self.blocks):
            x = block(x, rotary_cos_sin, c, seqlens=None)
            if layer_idx is not None and i == layer_idx:
                rep = x
        x = self.output_layer(x, c)

        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        return x, rep
