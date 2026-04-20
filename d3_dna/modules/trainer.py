"""
D3-DNA training.

Provides:
    D3LightningModule  -- PL LightningModule wrapping the score model + EMA + loss
    D3DataModule       -- PL LightningDataModule that accepts pre-built datasets
    D3Trainer          -- high-level orchestrator that builds a pl.Trainer

Training-mechanic factories (loss fn, optimizer, step fn) are co-located here
because they are called exclusively from D3LightningModule. They are not
"losses" in the evaluation-metric sense — see d3_dna/evals/metrics.py for that.
"""

import os
import datetime
from itertools import chain
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('medium')

from d3_dna.models.ema import ExponentialMovingAverage
from d3_dna.models import TransformerModel, ConvolutionalModel
from d3_dna.models.diffusion import get_graph, get_noise, get_score_fn
from d3_dna.modules.checkpoint import load_config


# =============================================================================
# TRAINING-MECHANIC FACTORIES
# =============================================================================

def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, sc=False):
    """Build the diffusion training-loss closure."""

    def loss_fn(model, batch, labels=None, cond=None, t=None, perturbed_batch=None):
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
            curr_sigma, curr_dsigma = noise(t / 2)
            curr_score = log_score_fn(perturbed_batch, curr_sigma, labels)
            t_dsigma = t / 2 * curr_dsigma
            rev_rate = t_dsigma[..., None, None] * graph.reverse_rate(perturbed_batch, curr_score)
            x = graph.sample_rate(perturbed_batch, rev_rate)

            sampling_eps_tensor = torch.tensor(sampling_eps, device=batch.device)
            next_sigma, next_dsigma = noise(sampling_eps_tensor)
            next_score = log_score_fn(x, next_sigma, labels)
            t_dsigma_next = sampling_eps_tensor * next_dsigma
            rev_rate_next = t_dsigma_next[..., None, None] * graph.reverse_rate(x, next_score)

            x_next = graph.sample_rate(x, rev_rate_next)
            l2_loss = ((batch - x_next) ** 2)
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
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2),
                                eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config` (lr warmup + grad clip)."""

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
    """Training step with gradient accumulation + EMA update."""
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


# =============================================================================
# LIGHTNING MODULES
# =============================================================================

class D3LightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for D3 DNA Discrete Diffusion.

    Config-driven: creates the model from config.model.architecture without
    requiring subclassing. For custom architectures, subclass and override
    create_model().
    """

    def __init__(self, cfg, dataset_name: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.dataset_name = dataset_name

        self.score_model = None
        self.graph = None
        self.noise = None
        self.ema = None
        self.loss_fn = None
        self.sampling_eps = 1e-5

        self.accum_iter = 0
        self.total_loss = 0

    def create_model(self):
        arch = self.cfg.model.architecture
        if arch == 'transformer':
            return TransformerModel(self.cfg)
        elif arch == 'convolutional':
            return ConvolutionalModel(self.cfg)
        else:
            raise ValueError(
                f"Unknown architecture '{arch}'. "
                "Expected 'transformer' or 'convolutional'. "
                "Subclass D3LightningModule and override create_model() for custom architectures."
            )

    def setup_ema(self):
        if self.score_model is not None:
            self.ema = ExponentialMovingAverage(
                self.score_model.parameters(),
                decay=self.cfg.training.ema
            )

    def setup(self, stage: str = None):
        if self.score_model is None:
            self.score_model = self.create_model()
            self.setup_ema()

        self.graph = get_graph(self.cfg, self.device)
        self.noise = get_noise(self.cfg).to(self.device)

        if hasattr(self, 'ema') and hasattr(self.ema, 'shadow_params'):
            for i, shadow_param in enumerate(self.ema.shadow_params):
                self.ema.shadow_params[i] = shadow_param.to(self.device)

        self.loss_fn = get_loss_fn(
            self.noise, self.graph, train=True, sampling_eps=self.sampling_eps
        )

    def process_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, target = batch
            return inputs, target
        else:
            return batch, None

    def training_step(self, batch, batch_idx):
        inputs, target = self.process_batch(batch)

        if target is not None:
            loss = self.loss_fn(self.score_model, inputs, target).mean()
        else:
            loss = self.loss_fn(self.score_model, inputs).mean()

        loss = loss / self.cfg.training.accum

        self.accum_iter += 1
        self.total_loss += loss.detach()

        if self.accum_iter == self.cfg.training.accum:
            self.accum_iter = 0
            if self.ema is not None:
                self.ema.update(self.score_model.parameters())

            accumulated_loss_log = self.total_loss
            self.log('train_loss', accumulated_loss_log, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.total_loss = 0

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = self.process_batch(batch)

        if self.ema is not None:
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model.parameters())

        eval_loss_fn = get_loss_fn(
            self.noise, self.graph, train=False, sampling_eps=self.sampling_eps
        )

        with torch.no_grad():
            if target is not None:
                loss = eval_loss_fn(self.score_model, inputs, target).mean()
            else:
                loss = eval_loss_fn(self.score_model, inputs).mean()

        if self.ema is not None:
            self.ema.restore(self.score_model.parameters())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = chain(self.score_model.parameters(), self.noise.parameters())
        optimizer = get_optimizer(self.cfg, params)

        if self.cfg.optim.warmup > 0:
            def lr_lambda(step):
                if step < self.cfg.optim.warmup:
                    return step / self.cfg.optim.warmup
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }

        return optimizer

    def on_before_optimizer_step(self, optimizer):
        if self.cfg.optim.grad_clip >= 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.cfg.optim.grad_clip,
                gradient_clip_algorithm="norm"
            )

    def load_from_original_checkpoint(self, checkpoint_path: str):
        loaded_state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = loaded_state.get('state_dict', loaded_state)

        model_dict = self.score_model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v
        model_dict.update(filtered_dict)
        self.score_model.load_state_dict(model_dict, strict=False)

        if 'ema' in loaded_state and self.ema is not None:
            self.ema.load_state_dict(loaded_state['ema'], device=self.device)

        return loaded_state.get('step', 0)

    def state_dict(self):
        state = super().state_dict()
        if hasattr(self, 'ema') and self.ema is not None:
            ema_state = self.ema.state_dict()
            for key, value in ema_state.items():
                state[f'ema.{key}'] = value
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        if 'model' in state_dict and 'ema' in state_dict and 'step' in state_dict:
            step = self.load_from_original_checkpoint(state_dict)
            return step
        else:
            model_state = {}
            ema_state = {}

            for key, value in state_dict.items():
                if key.startswith('ema.'):
                    ema_key = key.replace('ema.', '')
                    ema_state[ema_key] = value
                else:
                    model_state[key] = value

            result = super().load_state_dict(model_state, strict=False)

            if ema_state and hasattr(self, 'ema') and self.ema is not None:
                try:
                    self.ema.load_state_dict(ema_state, device=self.device)
                except Exception as e:
                    print(f"Could not load EMA state: {e}")

            return result


class D3DataModule(pl.LightningDataModule):
    """Concrete DataModule that accepts pre-built datasets directly."""

    def __init__(self, cfg, train_dataset, val_dataset):
        super().__init__()
        self.cfg = cfg
        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str = None):
        self.train_ds = self._train_ds
        self.val_ds = self._val_ds

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum),
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.eval.batch_size // (self.cfg.ngpus * self.cfg.training.accum),
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )


# =============================================================================
# HIGH-LEVEL TRAINER
# =============================================================================

class D3Trainer:
    """
    High-level trainer for D3 DNA Discrete Diffusion models.

    Example::

        trainer = D3Trainer('config.yaml')
        trainer.fit(train_dataset, val_dataset)
    """

    def __init__(self, config, work_dir: Optional[str] = None, callbacks=None):
        if isinstance(config, (str, os.PathLike)):
            config = load_config(str(config))
        self.cfg = config
        self.extra_callbacks = callbacks or []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = work_dir or f"experiments/d3_run_{timestamp}"

    def fit(self, train_dataset, val_dataset, resume_from: Optional[str] = None):
        os.makedirs(self.work_dir, exist_ok=True)

        lightning_module = D3LightningModule(self.cfg)
        data_module = D3DataModule(self.cfg, train_dataset, val_dataset)
        trainer = self._build_trainer()

        ckpt_path = None
        if resume_from and os.path.exists(resume_from):
            ckpt_path = resume_from

        trainer.fit(lightning_module, data_module, ckpt_path=ckpt_path)
        print(f"Training completed. Results saved to: {self.work_dir}")
        return trainer, lightning_module

    def _build_trainer(self):
        loggers = self._setup_logging()
        callbacks = self._setup_callbacks(has_logger=bool(loggers))

        trainer_args = {
            'max_epochs': self.cfg.training.get('max_epochs', 300),
            'log_every_n_steps': self.cfg.training.get('log_freq', 50),
            'check_val_every_n_epoch': self.cfg.training.get('val_every_n_epochs', 4),
            'accumulate_grad_batches': self.cfg.training.accum,
            # fp16-mixed matches the fp16 autocast in get_score_fn (d3_dna/models/
            # diffusion.py) that the training loss path also goes through. Lightning
            # installs a GradScaler for '16-mixed' (needed for fp16 gradient stability)
            # but not for 'bf16-mixed'.
            'precision': '16-mixed',
            'gradient_clip_val': self.cfg.optim.grad_clip if self.cfg.optim.grad_clip >= 0 else None,
            'enable_checkpointing': True,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'callbacks': callbacks,
            'logger': loggers,
        }

        if self.cfg.ngpus > 1:
            trainer_args.update({
                'devices': self.cfg.ngpus,
                'num_nodes': getattr(self.cfg, 'nnodes', 1),
                'strategy': 'ddp_find_unused_parameters_true',
                'sync_batchnorm': True,
            })
        else:
            trainer_args['devices'] = 1

        return pl.Trainer(**trainer_args)

    def _setup_callbacks(self, has_logger=True):
        callbacks = []

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.work_dir, "checkpoints"),
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
            every_n_epochs=self.cfg.training.get('checkpoint_every_n_epochs', 10)
        )
        callbacks.append(checkpoint_callback)
        if has_logger:
            callbacks.append(LearningRateMonitor(logging_interval='step'))

        if hasattr(self.cfg.training, 'early_stopping_patience') and self.cfg.training.early_stopping_patience:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.cfg.training.early_stopping_patience,
                mode='min'
            )
            callbacks.append(early_stop)

        callbacks.extend(self.extra_callbacks)
        return callbacks

    def _setup_logging(self):
        loggers = []

        try:
            tb_logger = TensorBoardLogger(
                save_dir=self.work_dir,
                name="lightning_logs",
                version=None
            )
            loggers.append(tb_logger)
        except (ImportError, ModuleNotFoundError):
            print("tensorboard not installed, skipping TensorBoard logging.")

        if hasattr(self.cfg, 'wandb') and self.cfg.wandb.get('enabled', False):
            try:
                from pytorch_lightning.loggers import WandbLogger
                config_dict = OmegaConf.to_container(self.cfg, resolve=True)
                wandb_logger = WandbLogger(
                    project=self.cfg.wandb.get('project', 'd3-dna'),
                    name=self.cfg.wandb.get('name', None),
                    entity=self.cfg.wandb.get('entity', None),
                    config=config_dict,
                    save_dir=self.work_dir,
                    id=self.cfg.wandb.get('id', None),
                )
                loggers.append(wandb_logger)
            except ImportError:
                print("wandb not installed. Install with: pip install d3-dna[logging]")

        return loggers


# Back-compat aliases for subclassing
BaseD3LightningModule = D3LightningModule
BaseD3DataModule = D3DataModule
