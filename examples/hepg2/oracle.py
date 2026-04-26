"""LentiMPRA oracle loader + LegNet model (vendored from the D3 repo).

Oracle for K562 lentiMPRA activity: predicts a single scalar per 230bp one-hot input.
Source architecture: D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/mpralegnet.py.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


INPUT_LEN = 230
N_FEATURES = 1


# ============================================================================
# LegNet architecture (vendored)
# ============================================================================

class SELayer(nn.Module):
    def __init__(self, inp, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp // reduction)), nn.SiLU(),
            nn.Linear(int(inp // reduction), inp), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class EffBlock(nn.Module):
    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = in_ch if out_ch is None else out_ch
        self.inner_dim = in_ch * resize_factor
        se_red = resize_factor if se_reduction is None else se_reduction
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, self.inner_dim, 1, padding="same", bias=False),
            nn.BatchNorm1d(self.inner_dim), activation(),
            nn.Conv1d(self.inner_dim, self.inner_dim, ks, groups=self.inner_dim, padding="same", bias=False),
            nn.BatchNorm1d(self.inner_dim), activation(),
            SELayer(self.inner_dim, reduction=se_red),
            nn.Conv1d(self.inner_dim, in_ch, 1, padding="same", bias=False),
            nn.BatchNorm1d(in_ch), activation(),
        )

    def forward(self, x):
        return self.block(x)


class LocalBlock(nn.Module):
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, padding="same", bias=False),
            nn.BatchNorm1d(out_ch), activation(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)


class MapperBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_features, out_features, 1),
        )

    def forward(self, x):
        return self.block(x)


class LegNet(nn.Module):
    def __init__(self, in_ch, stem_ch, stem_ks, ef_ks, ef_block_sizes, pool_sizes,
                 resize_factor, output_dim=1, activation=nn.SiLU):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)
        self.in_ch = in_ch
        self.output_dim = output_dim
        self.stem = LocalBlock(in_ch=in_ch, out_ch=stem_ch, ks=stem_ks, activation=activation)

        blocks = []
        in_c = stem_ch
        out_c = stem_ch
        for pool_sz, out_c in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(EffBlock(in_ch=in_c, out_ch=in_c, ks=ef_ks,
                                        resize_factor=resize_factor, activation=activation)),
                LocalBlock(in_ch=in_c * 2, out_ch=out_c, ks=ef_ks, activation=activation),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity(),
            )
            in_c = out_c
            blocks.append(blc)
        self.main = nn.Sequential(*blocks)
        self.mapper = MapperBlock(in_features=out_c, out_features=out_c * 2)
        self.head = nn.Sequential(
            nn.Linear(out_c * 2, out_c * 2),
            nn.BatchNorm1d(out_c * 2), activation(),
            nn.Linear(out_c * 2, output_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.head(x)
        if self.output_dim == 1:
            x = x.squeeze(-1)
        return x


# ============================================================================
# Minimal training-config (only fields needed to build the model)
# ============================================================================

@dataclass
class _LegNetConfig:
    stem_ch: int = 64
    stem_ks: int = 11
    ef_ks: int = 9
    ef_block_sizes: List[int] = field(default_factory=lambda: [80, 96, 112, 128])
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    resize_factor: int = 4
    use_reverse_channel: bool = False
    output_dim: int = 1

    @property
    def in_ch(self) -> int:
        return 4 + int(self.use_reverse_channel)

    def get_model(self) -> nn.Module:
        return LegNet(
            in_ch=self.in_ch, stem_ch=self.stem_ch, stem_ks=self.stem_ks, ef_ks=self.ef_ks,
            ef_block_sizes=self.ef_block_sizes, pool_sizes=self.pool_sizes,
            resize_factor=self.resize_factor, output_dim=self.output_dim,
        )

    @classmethod
    def from_json(cls, path: Union[Path, str]) -> "_LegNetConfig":
        with open(path) as f:
            raw = json.load(f)
        accepted = {k: raw[k] for k in (
            "stem_ch", "stem_ks", "ef_ks", "ef_block_sizes", "pool_sizes",
            "resize_factor", "use_reverse_channel", "output_dim",
        ) if k in raw}
        return cls(**accepted)


# ============================================================================
# Oracle wrapper + loader
# ============================================================================

class LentiMPRAOracle:
    """LegNet oracle: predicts (N, 1) activity from (N, 4, 230) one-hot."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, x, batch_size=256, progress=True):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != INPUT_LEN:
            raise ValueError(f"LegNet expects L={INPUT_LEN}, got {x.shape[-1]}")
        out = []
        it = range(0, len(x), batch_size)
        if progress:
            it = tqdm(it, desc="LegNet predict")
        for i in it:
            b = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=self.device)
            y = self.model(b)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            out.append(y.cpu().numpy())
        return np.concatenate(out, axis=0)  # (N, 1)


def _strip_prefix(sd, prefix="model."):
    pref_len = len(prefix)
    return {k[pref_len:] if k.startswith(prefix) else k: v for k, v in sd.items()}


def load(checkpoint_path, device, config_path=None):
    """Build LegNet, load weights, return an oracle wrapper.

    `config_path` is optional: if omitted, the dataclass defaults below are used,
    which match the architecture trained against the Zenodo Oracle_MPRA_K562.ckpt.
    Pass an explicit JSON path only if loading a non-default LegNet variant.
    """
    cfg = _LegNetConfig.from_json(config_path) if config_path else _LegNetConfig()
    model = cfg.get_model()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = _strip_prefix(sd, "model.")
    model.load_state_dict(sd, strict=False)
    return LentiMPRAOracle(model.to(device).eval(), device)
