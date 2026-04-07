"""
D3-DNA Diffusion Process

Merges noise schedules, graph/transition matrices, losses, and categorical sampling
into a single module.
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


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
    """
    Baseline forward method to get the total + rate of noise at a timestep.
    """
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
    """
    Compute the bridge interpolation functions alpha(t), beta(t), gamma(t).
    """
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
# LOSSES AND OPTIMIZERS
# =============================================================================

def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, sc=False):

    def loss_fn(model, batch, labels=None, cond=None, t=None, perturbed_batch=None):
        from d3_dna.io import get_score_fn  # lazy import to avoid circular dependency

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma, labels)

        if sc:
            curr_sigma, curr_dsigma = noise(t/2)
            curr_score = log_score_fn(perturbed_batch, curr_sigma, labels)
            t_dsigma = t/2 * curr_dsigma
            rev_rate = t_dsigma[..., None, None] * graph.reverse_rate(perturbed_batch, curr_score)
            x = graph.sample_rate(perturbed_batch, rev_rate)

            sampling_eps_tensor = torch.tensor(sampling_eps, device=batch.device)
            next_sigma, next_dsigma = noise(sampling_eps_tensor)
            next_score = log_score_fn(x, next_sigma, labels)
            t_dsigma_next = sampling_eps_tensor * next_dsigma
            rev_rate_next = t_dsigma_next[..., None, None] * graph.reverse_rate(x, next_score)

            x_next = graph.sample_rate(x, rev_rate_next)
            l2_loss = ((batch - x_next)**2)
            mask = torch.rand(batch.shape[0], device=batch.device) < 0.25
            expanded_mask = mask.unsqueeze(-1).expand_as(l2_loss)
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            loss = (dsigma[:, None] * loss)
            main_loss = loss.clone()
            main_loss[expanded_mask] = loss[expanded_mask] + l2_loss[expanded_mask]
            final_loss = main_loss.sum(dim=-1)
            return final_loss
        else:
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            loss = (dsigma[:, None] * loss).sum(dim=-1)
            return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, scaler, params, step,
                    lr=config.optim.lr, warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, labels=None, cond=None):
        nonlocal accum_iter
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, labels, cond=cond).mean() / accum

            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()

                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, labels, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn
