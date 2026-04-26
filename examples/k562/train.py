"""
Train a D3 model on K562 LentiMPRA sequences with SP-MSE validation.

Usage:
    python train.py --config config_transformer.yaml
    python train.py --config config_conv.yaml --output-dir outputs/k562_conv
    python train.py --config config_transformer.yaml \\
        --resume-from outputs/k562_transformer/checkpoints/last.ckpt

Importable:
    from train import main
    main("config_conv.yaml", output_dir="outputs/...", resume_from=None)
"""

import argparse
from typing import Optional

from omegaconf import OmegaConf

from d3_dna import D3Trainer
from data import K562Dataset, get_data_file, get_oracle_file
from callbacks import K562MSECallback


def main(
    config: str,
    output_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    data_file: Optional[str] = None,
    oracle_file: Optional[str] = None,
) -> None:
    cfg = OmegaConf.load(config)

    data_path = get_data_file(cfg, override=data_file)
    oracle_path = get_oracle_file(cfg, override=oracle_file)

    train_ds = K562Dataset(data_path, split="train")
    val_ds = K562Dataset(data_path, split="valid")
    print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    validation_samples = 500 if cfg.model.architecture == "transformer" else 1000
    sp_mse_callback = K562MSECallback(
        oracle_path=str(oracle_path),
        data_path=str(data_path),
        validation_freq_epochs=cfg.training.get("val_every_n_epochs", 4),
        validation_samples=validation_samples,
        sampling_steps=20,
    )

    work_dir = output_dir or f"train_experiments/k562_{cfg.model.architecture}"
    trainer = D3Trainer(cfg, work_dir=work_dir, callbacks=[sp_mse_callback])
    trainer.fit(train_ds, val_ds, resume_from=resume_from)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--data-file", default=None,
                   help="Override config; if absent, downloads from Zenodo.")
    p.add_argument("--oracle-file", default=None,
                   help="Override config; if absent, downloads from Zenodo.")
    args = p.parse_args()
    main(
        config=args.config,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        data_file=args.data_file,
        oracle_file=args.oracle_file,
    )
