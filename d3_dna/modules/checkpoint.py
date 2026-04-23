"""
Shared config + checkpoint loaders.

Used by D3Trainer, D3Sampler, and anything else that needs to consume either an
OmegaConf YAML or a trained checkpoint (supports both PL `.ckpt` format and the
original D3 `{'model', 'ema', 'step'}` format).
"""

import os
from typing import Any, Tuple

import torch
from omegaconf import OmegaConf


def load_config(config_path: str):
    """Load an OmegaConf config from a YAML file."""
    return OmegaConf.load(config_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    config,
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load a trained D3 model from a checkpoint file.

    Handles both:
      - PyTorch Lightning .ckpt files (state_dict with 'score_model.' prefix)
      - Original D3 format (keys: 'model', 'ema', 'step')

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: An instantiated (but untrained) model of the correct architecture.
        config: OmegaConf config used to build graph and noise.
        device: Device string ('cuda' or 'cpu').

    Returns:
        Tuple of (model, graph, noise) with EMA weights applied to model.
    """
    from d3_dna.models.ema import ExponentialMovingAverage
    from d3_dna.models.diffusion import get_graph, get_noise

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = model.to(device)
    graph = get_graph(config, device)
    noise = get_noise(config).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if checkpoint_path.endswith('.ckpt'):
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_state, ema_state = {}, {}
        for key, value in state_dict.items():
            if key.startswith('score_model.'):
                model_state[key.replace('score_model.', '')] = value
            elif key.startswith('model.'):
                model_state[key.replace('model.', '')] = value
            elif key.startswith('ema.'):
                ema_state[key.replace('ema.', '')] = value
        if model_state:
            model.load_state_dict(model_state, strict=False)
        if ema_state:
            ema.load_state_dict(ema_state)
    else:
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        if 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])

    ema.store(model.parameters())
    ema.copy_to(model.parameters())

    return model, graph, noise
