"""
D3-DNA Sampling

Predictor-corrector sampler, predictors (Euler, Analytic, Bridge), denoiser,
and the high-level D3Sampler class.
"""

import abc
import os
import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
from d3_dna.diffusion import sample_categorical
from d3_dna.io import get_score_fn


_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, labels, t, step_size):
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma, labels)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x


@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma, labels)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


@register_predictor(name="euler_bridge")
class EulerBridgePredictor:
    """Predictor for SE-DDB using Euler tau-leaping."""

    def __init__(self, R_fn):
        self.R_fn = R_fn

    def update_fn(self, score_fn, x, t, dt):
        guided_scores = score_fn(x, t)
        current_tokens_one_hot = F.one_hot(x, num_classes=guided_scores.shape[-1])
        base_rate = self.R_fn(x, t)
        transition_probs = current_tokens_one_hot + dt * base_rate * guided_scores
        transition_probs = torch.clamp(transition_probs, min=0)
        renorm_factor = transition_probs.sum(dim=-1, keepdim=True)
        transition_probs = transition_probs / renorm_factor.clamp(min=1e-9)
        B, L, V = transition_probs.shape
        return sample_categorical(transition_probs.view(B * L, V)).view(B, L)


@register_predictor(name="tweedie_bridge")
class TweedieBridgePredictor:
    """Predictor for SE-DDB using Tweedie tau-leaping."""

    def __init__(self, Q, noise_schedule):
        self.Q = Q
        self.noise_schedule = noise_schedule

    def update_fn(self, score_fn, x, t, dt):
        guided_scores = score_fn(x, t)
        s_curr = self.noise_schedule.total_noise(t.squeeze())
        s_prev = self.noise_schedule.total_noise((t - dt).squeeze())
        sigma_delta = s_curr - s_prev
        sigma_delta = sigma_delta.view(-1, 1, 1, 1)

        Q_device = self.Q.to(device=x.device, dtype=guided_scores.dtype)
        sigma_delta = sigma_delta.to(dtype=guided_scores.dtype)

        mat_forward = torch.matrix_exp(-sigma_delta * Q_device)
        mat_reverse = torch.matrix_exp(sigma_delta * Q_device)

        guided_scores_r = guided_scores.unsqueeze(-1)
        corrected_scores = torch.matmul(mat_forward, guided_scores_r)

        current_token_one_hot = F.one_hot(x, num_classes=Q_device.shape[0]).to(dtype=guided_scores.dtype)
        denoising_engine = torch.matmul(current_token_one_hot.unsqueeze(2), mat_reverse).squeeze(2)

        transition_probs = corrected_scores.squeeze(-1) * denoising_engine
        transition_probs = torch.clamp(transition_probs, min=0)
        renorm_factor = transition_probs.sum(dim=-1, keepdim=True)
        transition_probs = transition_probs / renorm_factor.clamp(min=1e-9)

        B, L, V = transition_probs.shape
        return sample_categorical(transition_probs.view(B * L, V)).view(B, L)


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, labels, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma, labels)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        if self.graph.absorb:
            probs = probs[..., :-1]

        return sample_categorical(probs)


def get_sampling_fn(config, graph, noise, batch_dims, eps, device, viz_logger=None):
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device,
                                 viz_logger=viz_logger)
    return sampling_fn


def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, viz_logger=None):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, labels):
        sampling_score_fn = get_score_fn(model, train=False, sampling=True)

        x = graph.sample_limit(*batch_dims).to(device)

        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in tqdm(range(steps), desc="Sampling", unit="step"):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)

            x = predictor.update_fn(sampling_score_fn, x, labels, t, dt)

        if denoise:
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, labels, t)

        return x

    return pc_sampler


def get_guided_bridge_sampler(
    predictor_name="euler_bridge",
    steps=128,
    device='cuda',
    graph=None,
    noise=None,
    Q=None,
    eps=1e-5,
    viz_logger=None
):
    """Creates a sampling function for the guided discrete diffusion bridge."""
    if predictor_name == "euler_bridge":
        def uniform_R_fn(x, t):
            return 1.0
        pred = EulerBridgePredictor(R_fn=uniform_R_fn)
    elif predictor_name == "tweedie_bridge":
        if Q is None and hasattr(graph, 'Q_matrix'):
            Q = graph.Q_matrix.to(device)
        elif Q is None:
            vocab_size = graph.dim
            Q = torch.ones(vocab_size, vocab_size, device=device) / vocab_size
            Q.fill_diagonal_(0)
            for i in range(vocab_size):
                Q[i, i] = -(vocab_size - 1) / vocab_size
        else:
            Q = Q.to(device)
        pred = TweedieBridgePredictor(Q=Q, noise_schedule=noise)
    else:
        raise ValueError(f"Predictor {predictor_name} not recognized.")

    @torch.no_grad()
    def guided_sampler(model, start_sequence_xT, target_label):
        x_t = start_sequence_xT.to(device)
        B, L = x_t.shape

        if isinstance(target_label, (int, float)):
            labels = torch.tensor([target_label] * B, device=device, dtype=torch.long)
        else:
            labels = target_label.to(device)

        timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
        dt = (1.0 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(B, device=device)

            def score_fn(x_current, time_current):
                paired_input = torch.stack([x_current, start_sequence_xT.to(device)], dim=1)
                sampling_score_fn = get_score_fn(model, train=False, sampling=True)
                scores = sampling_score_fn(paired_input, time_current, labels)
                return scores

            x_t = pred.update_fn(score_fn, x_t, t, dt)

        return x_t

    return guided_sampler


# =============================================================================
# D3Sampler — High-level API
# =============================================================================

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
    """

    def __init__(self, config):
        if isinstance(config, (str, os.PathLike)):
            from d3_dna.io import load_config
            config = load_config(config)
        self.config = config

    def generate(
        self,
        checkpoint: str,
        model: torch.nn.Module,
        num_samples: int,
        labels=None,
        steps: Optional[int] = None,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate sequences using the PC sampler.

        Args:
            checkpoint: Path to trained model checkpoint.
            model: Model instance of the correct architecture (will be loaded from ckpt).
            num_samples: Number of sequences to generate.
            labels: Optional conditioning labels, shape (num_samples, signal_dim).
            steps: Sampling steps; defaults to config.sampling.steps.
            device: Device to run on.

        Returns:
            Integer token tensor of shape (num_samples, sequence_length).
        """
        from d3_dna.io import load_checkpoint

        print(f"Loading checkpoint: {checkpoint}")
        model, graph, noise = load_checkpoint(checkpoint, model, self.config, device)
        model.eval()

        seq_len = self.config.dataset.sequence_length
        n_steps = steps if steps is not None else self.config.sampling.steps
        print(f"Generating {num_samples} sequences (len={seq_len}, steps={n_steps}, device={device})")

        sampling_fn = get_pc_sampler(
            graph=graph,
            noise=noise,
            batch_dims=(num_samples, seq_len),
            predictor=self.config.sampling.predictor,
            steps=n_steps,
            denoise=self.config.sampling.noise_removal,
            device=torch.device(device),
        )

        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=device)
        elif labels is not None:
            labels = labels.to(device)

        sequences = sampling_fn(model, labels)
        return sequences

    def save(self, sequences: torch.Tensor, path: str, format: str = 'fasta'):
        """
        Save generated sequences to a file.

        Args:
            sequences: Integer token tensor (num_samples, seq_len).
            path: Output file path.
            format: 'fasta', 'txt', 'npz', 'h5', or 'csv'.
        """
        from d3_dna.io import sequences_to_strings

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
