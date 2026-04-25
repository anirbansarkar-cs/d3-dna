"""
D3-DNA diffusion + sampling math.

Pure PyTorch (zero Lightning imports). Contains everything needed to define the
forward diffusion process and to run reverse sampling:

    * Categorical sampling utilities (``gumbel_softmax``, ``sample_categorical``)
    * Noise schedules (``Noise``, ``GeometricNoise``, ``get_noise``)
    * Graphs / transition matrices (``Graph``, ``Uniform``, ``Absorbing``,
      ``Bridge``, ``get_graph``, ``bridge_interpolation_functions``)
    * Score-model wrappers (``get_model_fn``, ``get_score_fn``)
    * Predictors (``Predictor``, ``EulerPredictor``, ``NonePredictor``,
      ``AnalyticPredictor``, ``EulerBridgePredictor``, ``TweedieBridgePredictor``,
      ``register_predictor``, ``get_predictor``)
    * Sampler factories (``Denoiser``, ``get_sampling_fn``, ``get_pc_sampler``,
      ``get_guided_bridge_sampler``)
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# CATEGORICAL SAMPLING UTILITIES
# =============================================================================

def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


# =============================================================================
# NOISE SCHEDULES
# =============================================================================

def get_noise(config):
    if config.noise.type == "geometric":
        return GeometricNoise(config.noise.sigma_min, config.noise.sigma_max)
    else:
        raise ValueError(f"{config.noise.type} is not a valid noise")


class Noise(abc.ABC, nn.Module):
    """Baseline forward method to get the total + rate of noise at a timestep."""

    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """Rate of change of noise ie g(t)"""
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        r"""Total noise ie \int_0^t g(t) dt + g(0)"""
        pass


class GeometricNoise(Noise, nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


# =============================================================================
# GRAPH / TRANSITION MATRICES
# =============================================================================

def bridge_interpolation_functions(t, T=1.0):
    """Compute the bridge interpolation functions alpha(t), beta(t), gamma(t)."""
    t_norm = t / T
    alpha = (1 - t_norm) ** 2
    beta = t_norm ** 2
    gamma = 2 * t_norm * (1 - t_norm)
    return alpha, beta, gamma


def get_graph(config, device):
    if config.graph.type == "uniform":
        return Uniform(config.tokens)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens)
    elif config.graph.type == "bridge":
        return Bridge(config.tokens)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        pass

    @abc.abstractmethod
    def rate(self, i):
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        pass

    def sample_transition(self, i, sigma):
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        normalized_rate = self.transp_rate(i) * score
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        pass


class Uniform(Graph):
    """Everything goes to everything else. Normalized down by dimension to avoid blowup."""

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim
        )

        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1

    @property
    def absorb(self):
        return True

    def rate(self, i):
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        return self.transp_transition(i, sigma)

    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert

    def staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy


class Bridge(Graph):
    """Bridge process that interpolates between start and end tokens with base distribution."""

    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        self._Q_matrix = torch.ones(dim, dim) / dim
        self._Q_matrix.fill_diagonal_(0)
        for i in range(dim):
            self._Q_matrix[i, i] = -(dim - 1) / dim

    @property
    def dim(self):
        return self._dim

    @property
    def Q_matrix(self):
        return self._Q_matrix

    @property
    def absorb(self):
        return False

    def bridge_transition(self, x0, xT, t, T=1.0):
        B, L = x0.shape
        vocab_size = self.dim
        device = x0.device

        alpha_t = (1 - t) ** 2
        beta_t = t ** 2
        gamma_t = 2 * t * (1 - t)

        alpha_t_r = alpha_t.view(B, 1, 1)
        beta_t_r = beta_t.view(B, 1, 1)
        gamma_t_r = gamma_t.view(B, 1, 1)

        p_base = torch.full((vocab_size,), 1.0 / vocab_size, device=device)
        p_base_r = p_base.view(1, 1, vocab_size)

        p_itk = (
            alpha_t_r * F.one_hot(x0, num_classes=vocab_size).float() +
            beta_t_r * F.one_hot(xT, num_classes=vocab_size).float() +
            gamma_t_r * p_base_r
        )

        return p_itk

    def sample_bridge_transition(self, x0, xT, t, T=1.0):
        p_itk = self.bridge_transition(x0, xT, t, T)
        B, L, vocab_size = p_itk.shape
        perturbed_batch = sample_categorical(p_itk.view(B * L, vocab_size)).view(B, L)
        return perturbed_batch

    def rate(self, i):
        return torch.zeros(*i.shape, self.dim, device=i.device)

    def transp_rate(self, i):
        return torch.zeros(*i.shape, self.dim, device=i.device)

    def transition(self, i, sigma):
        return torch.zeros(*i.shape, self.dim, device=i.device)

    def sample_transition(self, i, sigma):
        return torch.randint_like(i, self.dim)

    def staggered_score(self, score, dsigma):
        return score

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        return torch.zeros_like(x, dtype=torch.float)

    def bridge_score_entropy(self, score, t, x_t, x_0, x_T):
        if t.dim() == 3:
            t_flat = t.squeeze(-1).squeeze(-1)
        elif t.dim() == 2:
            t_flat = t.squeeze(-1)
        else:
            t_flat = t

        alpha_t, beta_t, gamma_t = bridge_interpolation_functions(t_flat, T=1.0)
        p_itk = self.bridge_transition(x_0, x_T, t_flat, T=1.0)

        p_it_xt = torch.gather(p_itk, -1, x_t.unsqueeze(-1))
        target_ratio = p_itk / p_it_xt.clamp(min=1e-9)

        predicted_score = score.exp()
        loss_terms = predicted_score - target_ratio * score

        mask = torch.ones_like(loss_terms).scatter_(
            -1, x_t.unsqueeze(-1), 0.
        )

        loss_per_sample = (loss_terms * mask).sum(dim=(-1, -2))

        return loss_per_sample, gamma_t

    def bridge_ratio(self, x_t_token, y_token, x_0_token, x_T_token, t, T=1.0):
        alpha, beta, gamma = bridge_interpolation_functions(t, T)

        q_xt = torch.zeros_like(alpha)
        q_xt += gamma / self.dim
        q_xt += alpha * (x_t_token == x_0_token).float()
        q_xt += beta * (x_t_token == x_T_token).float()

        q_y = torch.zeros_like(alpha)
        q_y += gamma / self.dim
        q_y += alpha * (y_token == x_0_token).float()
        q_y += beta * (y_token == x_T_token).float()

        ratio = q_y / (q_xt + 1e-8)
        return ratio


# =============================================================================
# SCORE-MODEL WRAPPERS
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


def get_score_fn(model, train=False, sampling=False, dtype=torch.float16):
    """Create a score function wrapper around the model.

    ``dtype`` controls the autocast dtype on CUDA. Default is fp16 (legacy
    behavior); pass ``torch.bfloat16`` for transformer architectures where
    bf16 is preferred for stability without a GradScaler.
    """
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, sigma, labels=None):
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type, dtype=dtype, enabled=(device_type == 'cuda')):
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
# PREDICTORS
# =============================================================================

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


# =============================================================================
# SAMPLER FACTORIES
# =============================================================================

def get_sampling_fn(config, graph, noise, batch_dims, eps, device, viz_logger=None, dtype=torch.float16):
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device,
                                 viz_logger=viz_logger,
                                 dtype=dtype)
    return sampling_fn


def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, viz_logger=None, dtype=torch.float16):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, labels):
        sampling_score_fn = get_score_fn(model, train=False, sampling=True, dtype=dtype)

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
    viz_logger=None,
    dtype=torch.float16,
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
                sampling_score_fn = get_score_fn(model, train=False, sampling=True, dtype=dtype)
                scores = sampling_score_fn(paired_input, time_current, labels)
                return scores

            x_t = pred.update_fn(score_fn, x_t, t, dt)

        return x_t

    return guided_sampler
