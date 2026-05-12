"""
Template — train.py for a new D3 example.

The body of main() is what you actually need; the if __name__ block is just
the standard CLI shim. Reference: examples/k562/train.py.
"""

import argparse
from typing import Optional

from omegaconf import OmegaConf

from d3_dna import D3Trainer
from data import MinimalDataset, get_data_file, get_oracle_file
from callbacks import MinimalSPMSECallback


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

    train_ds = MinimalDataset(data_path, split="train")
    val_ds = MinimalDataset(data_path, split="valid")
    print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")

    sp_mse_callback = MinimalSPMSECallback(
        oracle_path=str(oracle_path),
        data_path=str(data_path),
        validation_freq_epochs=cfg.training.get("val_every_n_epochs", 4),
        validation_samples=500 if cfg.model.architecture == "transformer" else 1000,
        sampling_steps=20,
    )

    work_dir = output_dir or f"train_experiments/minimal_{cfg.model.architecture}"
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
