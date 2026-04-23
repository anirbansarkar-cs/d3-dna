"""
Train a D3 model on promoter sequences with SP-MSE validation.

Usage:
    python train.py --config config_transformer.yaml
    python train.py --config config_conv.yaml --work-dir outputs/promoter_conv
    python train.py --config config_transformer.yaml --resume outputs/promoter_tran/checkpoints/last.ckpt
"""

import argparse

from omegaconf import OmegaConf

from d3_dna import D3Trainer
from data import PromoterDataset
from callbacks import PromoterSPMSECallback

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_transformer.yaml")
parser.add_argument("--work-dir", type=str, default=None)
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()

cfg = OmegaConf.load(args.config)

train_ds = PromoterDataset(cfg.paths.data_file, split="train")
val_ds = PromoterDataset(cfg.paths.data_file, split="valid")
print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

validation_samples = 500 if cfg.model.architecture == "transformer" else 1000
sp_mse_callback = PromoterSPMSECallback(
    oracle_path=cfg.paths.oracle_model,
    data_path=cfg.paths.data_file,
    validation_freq_epochs=cfg.training.get("val_every_n_epochs", 4),
    validation_samples=validation_samples,
    sampling_steps=128,
)

work_dir = args.work_dir or f"outputs/promoter_{cfg.model.architecture}"
trainer = D3Trainer(cfg, work_dir=work_dir, callbacks=[sp_mse_callback])
trainer.fit(train_ds, val_ds, resume_from=args.resume)
