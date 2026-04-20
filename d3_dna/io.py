"""
D3-DNA I/O Utilities

Checkpoint loading/saving, config loading, data utilities, and sequence I/O.
"""

import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Any, List, Dict
from omegaconf import OmegaConf


# =============================================================================
# CONFIG UTILITIES
# =============================================================================

def load_config(config_path: str):
    """Load an OmegaConf config from a YAML file."""
    return OmegaConf.load(config_path)


def update_cfg_with_unknown_args(cfg, unknown_args):
    """Update cfg with command-line arguments not parsed by argparse."""
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg[2:]
            if (i + 1) < len(unknown_args):
                value = unknown_args[i + 1]
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                OmegaConf.update(cfg, key, value, merge=True)
                i += 2
            else:
                i += 1
        else:
            i += 1


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model."""

    def model_fn(x, sigma, labels=None):
        if train:
            model.train()
        else:
            model.eval()
        return model(x, labels, train, sigma)

    return model_fn


def get_score_fn(model, train=False, sampling=False):
    """Create a score function wrapper around the model."""
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, sigma, labels=None):
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
            sigma = sigma.reshape(-1)
            model_output = model_fn(x, sigma, labels)
            if isinstance(model_output, tuple):
                score, _ = model_output
            else:
                score = model_output

            if sampling:
                return score.exp()

            return score

    return score_fn


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

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
    from d3_dna.diffusion import get_graph, get_noise

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = model.to(device)
    graph = get_graph(config, device)
    noise = get_noise(config).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if checkpoint_path.endswith('.ckpt'):
        # Lightning format
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
        # Original D3 format
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        if 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])

    ema.store(model.parameters())
    ema.copy_to(model.parameters())

    return model, graph, noise


def restore_checkpoint(ckpt_dir, state, device):
    """Restore a full training state from checkpoint."""
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    """Save a full training state to checkpoint."""
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


# =============================================================================
# LOGGING
# =============================================================================

def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


# =============================================================================
# DATA UTILITIES
# =============================================================================

def cycle_loader(dataloader, sampler=None):
    """Create an infinite iterator from a DataLoader."""
    while True:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def collate_fn_generic(batch):
    """Generic collate function for batching data."""
    if isinstance(batch[0], (list, tuple)):
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        inputs = torch.stack(inputs) if isinstance(inputs[0], torch.Tensor) else torch.tensor(inputs)
        targets = torch.stack(targets) if isinstance(targets[0], torch.Tensor) else torch.tensor(targets)
        return inputs, targets
    else:
        return torch.stack(batch) if isinstance(batch[0], torch.Tensor) else torch.tensor(batch)


def compute_dataset_stats(dataset):
    """Compute basic statistics for a dataset."""
    stats = {
        'num_samples': len(dataset),
        'sample_shape': None,
        'data_type': None
    }
    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, (list, tuple)):
            stats['sample_shape'] = [item.shape if hasattr(item, 'shape') else len(item) for item in sample]
            stats['data_type'] = [type(item).__name__ for item in sample]
        else:
            stats['sample_shape'] = sample.shape if hasattr(sample, 'shape') else len(sample)
            stats['data_type'] = type(sample).__name__
    return stats


def create_sequence_mask(sequences, pad_token=None):
    """Create attention mask for sequences."""
    if pad_token is None:
        return torch.ones_like(sequences, dtype=torch.bool)
    else:
        return sequences != pad_token


def one_hot_encode_sequences(sequences, num_classes=4):
    """Convert token sequences to one-hot encoding."""
    return F.one_hot(sequences, num_classes=num_classes).float()


def sequences_to_strings(sequences, token_to_char=None):
    """Convert token sequences to strings."""
    if token_to_char is None:
        token_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    sequences_str = []
    for seq in sequences:
        seq_str = ''.join([token_to_char.get(token.item(), 'N') for token in seq])
        sequences_str.append(seq_str)
    return sequences_str


def calculate_gc_content(sequences):
    """Calculate GC content for DNA sequences (0=A, 1=C, 2=G, 3=T)."""
    gc_counts = ((sequences == 1) | (sequences == 2)).sum(dim=1).float()
    total_length = sequences.shape[1]
    return gc_counts / total_length


def reverse_complement(sequences):
    """Generate reverse complement of DNA sequences (0=A, 1=C, 2=G, 3=T)."""
    complement_map = torch.tensor([3, 2, 1, 0], device=sequences.device)
    complement_sequences = complement_map[sequences]
    reverse_complement_sequences = torch.flip(complement_sequences, dims=[1])
    return reverse_complement_sequences
