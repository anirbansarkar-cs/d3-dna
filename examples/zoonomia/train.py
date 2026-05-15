"""
Train a D3 model on Zoonomia cCRE windows (human-only, unconditional).

ZoonomiaDataset yields X: (L, 5) float one-hot and y: (1, 2) float
[class_idx, species_idx]. The package's score model needs X as a LongTensor
of token ids. A thin adapter does ``X.argmax(-1).long()`` and *drops* y so
DataLoader collation produces a bare tensor batch. The default
``D3LightningModule.process_batch`` then falls through to ``(batch, None)``
and the loss is computed without labels — unconditional training.

Config defaults (``config_{transformer,conv}.yaml``) already set
``dataset.signal_dim: 0`` and enable wandb at ``d3-cshl / d3-dna-ccre``, so
this script ships no runtime overrides — only the dataset-shape adapter and
an optional ``--ngpus`` for DDP.

Tokens are kept at 5 (N, A, C, G, T). The N row is dead weight for
species_index=0 training (verified: 0 Ns across 2k random windows). Mask
the logit for token 0 at sampling time if zero-N generation is required.

Multi-GPU / DDP
---------------
Pass ``--ngpus N`` (or edit ``cfg.ngpus`` in the YAML). D3Trainer drives
Lightning with strategy='ddp_find_unused_parameters_true' and
sync_batchnorm=True when ngpus > 1. The global batch in the YAML is split
per rank as ``batch_size // (ngpus * accum)``.

Usage
-----
    python train.py --config config_transformer.yaml
    python train.py --config config_transformer.yaml --ngpus 4
    python train.py --config config_conv.yaml --output-dir outputs/zoonomia_conv
    python train.py --config config_transformer.yaml \\
        --resume-from outputs/zoonomia_transformer/checkpoints/last.ckpt
"""

# DataLoader workers must use 'spawn' (not fork) under DDP — forked workers
# inherit the rank's CUDA state and the pin_memory thread inside the child
# aborts with SIGABRT. The package hardcodes pin_memory=True and we are not
# editing the package, so we set the default mp start method globally here.
# This MUST run before any import that materializes a multiprocessing state.
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
from typing import Optional

import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf, open_dict

from d3_dna import D3Trainer
from data import ZoonomiaDataset


class _ZoonomiaUnconditionalAdapter(Dataset):
    """Argmax one-hot X to token ids and drop y.

    Returns a bare ``LongTensor[L]`` so DataLoader yields a single tensor
    batch. ``D3LightningModule.process_batch`` then routes it as
    ``(batch, None)`` and the loss runs without labels.
    """

    def __init__(self, base: ZoonomiaDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> torch.Tensor:
        X, _ = self.base[idx]
        return X.argmax(dim=-1).long()


def _dataset_kwargs(cfg) -> dict:
    kwargs: dict = {"sequence_length": int(cfg.dataset.sequence_length)}
    data_cfg = cfg.get("data", None)
    if data_cfg is not None:
        if "h5_path" in data_cfg:
            kwargs["h5_path"] = str(data_cfg.h5_path)
        if "bed_path" in data_cfg:
            kwargs["bed_path"] = str(data_cfg.bed_path)
        if "species_index" in data_cfg:
            kwargs["species_index"] = int(data_cfg.species_index)
    return kwargs


def main(
    config: str,
    output_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    ngpus: Optional[int] = None,
) -> None:
    cfg = OmegaConf.load(config)
    if ngpus is not None:
        with open_dict(cfg):
            cfg.ngpus = int(ngpus)

    ds_kwargs = _dataset_kwargs(cfg)
    train_ds = _ZoonomiaUnconditionalAdapter(ZoonomiaDataset(split="train", **ds_kwargs))
    val_ds = _ZoonomiaUnconditionalAdapter(ZoonomiaDataset(split="val", **ds_kwargs))

    accum = int(cfg.training.get("accum", 1))
    per_rank_bs = int(cfg.training.batch_size) // (int(cfg.ngpus) * accum)
    wandb_cfg = cfg.get("wandb", OmegaConf.create({"enabled": False}))
    print(f"Train: {len(train_ds):,} sequences, Val: {len(val_ds):,} sequences")
    print(
        f"ngpus={cfg.ngpus} nnodes={cfg.get('nnodes', 1)} "
        f"global_batch={cfg.training.batch_size} per_rank_batch={per_rank_bs} accum={accum}"
    )
    print(
        f"unconditional (signal_dim={cfg.dataset.signal_dim}); "
        f"wandb={'on' if wandb_cfg.get('enabled', False) else 'off'} "
        f"({wandb_cfg.get('entity', '?')}/{wandb_cfg.get('project', '?')})"
    )

    work_dir = output_dir or f"train_experiments/zoonomia_{cfg.model.architecture}"
    trainer = D3Trainer(cfg, work_dir=work_dir)
    trainer.fit(train_ds, val_ds, resume_from=resume_from)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument(
        "--ngpus", type=int, default=None,
        help="Override cfg.ngpus (set >1 for DDP).",
    )
    args = p.parse_args()
    main(
        config=args.config,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        ngpus=args.ngpus,
    )
