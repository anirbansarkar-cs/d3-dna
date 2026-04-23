"""
Train a D3 transformer on K562 MPRA sequences with SP-MSE validation.

Usage:
    python train.py
    python train.py --resume outputs/k562/checkpoints/model-epoch=100-val_loss=260.ckpt
"""

import argparse
from omegaconf import OmegaConf
from d3_dna import D3Trainer
from data import K562Dataset
from callbacks import K562MSECallback

cfg = OmegaConf.load("config.yaml")

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
args = parser.parse_args()

train_ds = K562Dataset(cfg.paths.data_file, split="train")
val_ds = K562Dataset(cfg.paths.data_file, split="valid")

print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

sp_mse_callback = K562MSECallback(
    oracle_path=cfg.paths.oracle_model,
    data_path=cfg.paths.data_file,
    validation_freq_epochs=cfg.training.get("val_every_n_epochs", 4),
    validation_samples=500,
    sampling_steps=20,
)

trainer = D3Trainer(cfg, work_dir="outputs/k562", callbacks=[sp_mse_callback])
trainer.fit(train_ds, val_ds, resume_from=args.resume)

print("Training complete. Checkpoints saved to outputs/k562/checkpoints/")
