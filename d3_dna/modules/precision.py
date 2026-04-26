"""Precision policy: pick autocast/Lightning precision per architecture.

Default policy:
    transformer   -> bf16-mixed / torch.bfloat16  (no GradScaler needed)
    convolutional -> 16-mixed   / torch.float16   (Lightning installs GradScaler)

Override by setting ``cfg.training.precision`` to ``'16-mixed'`` or
``'bf16-mixed'`` (or ``'32-true'`` for fp32 debugging).
"""

from typing import Tuple

import torch

_VALID = {
    "16-mixed": torch.float16,
    "bf16-mixed": torch.bfloat16,
    "32-true": torch.float32,
}


def precision_for_cfg(cfg) -> Tuple[str, torch.dtype]:
    """Return the (Lightning precision string, autocast dtype) pair."""
    arch = getattr(cfg.model, "architecture", "transformer")
    default = "bf16-mixed" if arch == "transformer" else "16-mixed"
    pl_str = getattr(cfg.training, "precision", None) or default
    if pl_str not in _VALID:
        raise ValueError(
            f"Unsupported precision '{pl_str}'. Expected one of {sorted(_VALID)}."
        )
    return pl_str, _VALID[pl_str]


def autocast_dtype_for_cfg(cfg) -> torch.dtype:
    """Just the autocast dtype — convenience for non-trainer call sites."""
    return precision_for_cfg(cfg)[1]
