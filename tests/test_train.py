"""Minimal training smoke test: ensure D3Trainer.fit completes end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


def test_minimal_train(tmp_path: Path, repo_root: Path, example_sys_path):
    """Run examples/minimal for 2 epochs on synthetic data.

    Asserts: training completes without raising, a checkpoint is written,
    and the final train loss is finite.
    """
    from d3_dna import D3Trainer

    # Load the minimal config; override paths that make no sense in a temp dir.
    with example_sys_path("minimal"):
        from data import make_synthetic_dataset
        cfg = OmegaConf.load(repo_root / "examples" / "minimal" / "config.yaml")
        cfg.training.max_epochs = 2
        cfg.training.batch_size = 32
        cfg.training.accum = 1
        cfg.wandb.enabled = False

        train_ds = make_synthetic_dataset(n_samples=256, seq_len=cfg.dataset.sequence_length)
        val_ds = make_synthetic_dataset(n_samples=64, seq_len=cfg.dataset.sequence_length)

        d3 = D3Trainer(cfg, work_dir=str(tmp_path))
        pl_trainer, pl_module = d3.fit(train_ds, val_ds)

    # A checkpoint must have landed in the work dir.
    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert ckpts, f"No checkpoints written to {tmp_path / 'checkpoints'}"

    # Train loss should be finite after 2 epochs.
    logged = pl_trainer.callback_metrics
    train_loss = logged.get("train_loss_epoch") or logged.get("train_loss")
    assert train_loss is not None, f"No train_loss in callback metrics: {dict(logged)}"
    assert train_loss.isfinite().item(), f"train_loss not finite: {train_loss}"
