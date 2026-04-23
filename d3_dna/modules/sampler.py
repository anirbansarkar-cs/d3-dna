"""
D3-DNA sampling API.

The user-facing D3Sampler class. Pure-math predictors and sampler factories live
in d3_dna.models.diffusion.
"""

import os
from typing import Optional

import torch

from d3_dna.models.diffusion import get_pc_sampler
from d3_dna.modules.checkpoint import load_config, load_checkpoint
from d3_dna.utils.dna import sequences_to_strings


class D3Sampler:
    """
    High-level interface for generating DNA sequences using a trained D3 model.

    Example::

        sampler = D3Sampler('config.yaml')
        model = TransformerModel(cfg)
        sequences = sampler.generate(
            checkpoint='path/to/model.ckpt',
            model=model,
            num_samples=100,
            labels=label_tensor
        )
        sampler.save(sequences, 'output.fasta')

    For large jobs, load the checkpoint once and generate in batches::

        sampler = D3Sampler('config.yaml')
        model = TransformerModel(cfg)
        sampler.load(checkpoint='path/to/model.ckpt', model=model, device='cuda')
        sequences = sampler.generate_batched(
            num_samples=7000,
            labels=label_tensor,
            batch_size=64,
        )
    """

    def __init__(self, config):
        if isinstance(config, (str, os.PathLike)):
            config = load_config(config)
        self.config = config
        self._model = None
        self._graph = None
        self._noise = None
        self._device = None

    def load(
        self,
        checkpoint: str,
        model: torch.nn.Module,
        device: str = 'cuda',
    ):
        """Load a checkpoint once for repeated generation calls."""
        print(f"Loading checkpoint: {checkpoint}")
        self._model, self._graph, self._noise = load_checkpoint(
            checkpoint, model, self.config, device
        )
        self._model.eval()
        self._device = device

    @property
    def is_loaded(self):
        return self._model is not None

    def _ensure_loaded(self, checkpoint, model, device):
        if not self.is_loaded:
            self.load(checkpoint, model, device)

    def generate(
        self,
        checkpoint: str,
        model: torch.nn.Module,
        num_samples: int,
        labels=None,
        steps: Optional[int] = None,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """Generate sequences using the PC sampler (loads checkpoint if needed)."""
        self._ensure_loaded(checkpoint, model, device)

        seq_len = self.config.dataset.sequence_length
        n_steps = steps if steps is not None else self.config.sampling.steps

        sampling_fn = get_pc_sampler(
            graph=self._graph,
            noise=self._noise,
            batch_dims=(num_samples, seq_len),
            predictor=self.config.sampling.predictor,
            steps=n_steps,
            denoise=self.config.sampling.noise_removal,
            device=torch.device(self._device),
        )

        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=self._device)
        elif labels is not None:
            labels = labels.to(self._device)

        sequences = sampling_fn(self._model, labels)
        return sequences

    def generate_batched(
        self,
        num_samples: int,
        labels=None,
        batch_size: int = 64,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate sequences in batches; requires a prior :meth:`load` call."""
        if not self.is_loaded:
            raise RuntimeError("Call sampler.load(checkpoint, model, device) before generate_batched()")

        seq_len = self.config.dataset.sequence_length
        n_steps = steps if steps is not None else self.config.sampling.steps
        device = torch.device(self._device)
        print(f"Generating {num_samples} sequences in batches of {batch_size} (len={seq_len}, steps={n_steps})")

        all_seqs = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            bs = end - start

            sampling_fn = get_pc_sampler(
                graph=self._graph,
                noise=self._noise,
                batch_dims=(bs, seq_len),
                predictor=self.config.sampling.predictor,
                steps=n_steps,
                denoise=self.config.sampling.noise_removal,
                device=device,
            )

            batch_labels = None
            if labels is not None:
                batch_labels = labels[start:end]
                if not isinstance(batch_labels, torch.Tensor):
                    batch_labels = torch.tensor(batch_labels, device=device)
                else:
                    batch_labels = batch_labels.to(device)

            batch_seqs = sampling_fn(self._model, batch_labels)
            all_seqs.append(batch_seqs.cpu())

        return torch.cat(all_seqs, dim=0)

    def save(self, sequences: torch.Tensor, path: str, format: str = 'fasta'):
        """Save generated sequences to a file (fasta/txt/npz/h5/csv)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        if format in ('fasta', 'txt'):
            seq_strings = sequences_to_strings(sequences.cpu())
            with open(path, 'w') as f:
                if format == 'fasta':
                    for i, s in enumerate(seq_strings):
                        f.write(f'>seq_{i}\n{s}\n')
                else:
                    for s in seq_strings:
                        f.write(s + '\n')
        elif format == 'npz':
            import numpy as np
            np.savez(path, sequences.cpu().numpy())
        elif format in ('h5', 'hdf5'):
            import h5py
            with h5py.File(path, 'w') as f:
                f.create_dataset('sequences', data=sequences.cpu().numpy())
        elif format == 'csv':
            seq_strings = sequences_to_strings(sequences.cpu())
            with open(path, 'w') as f:
                f.write("sequence_id,sequence\n")
                for i, s in enumerate(seq_strings):
                    f.write(f"seq_{i},{s}\n")
        else:
            raise ValueError(f"Unsupported format: {format}")
