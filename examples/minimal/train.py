"""Train a D3 transformer on synthetic DNA data."""
from d3_dna import D3Trainer
from data import make_synthetic_dataset

train_ds = make_synthetic_dataset(n_samples=1000, seq_len=64)
val_ds = make_synthetic_dataset(n_samples=200, seq_len=64)

trainer = D3Trainer('config.yaml', work_dir='outputs/minimal_run')
trainer.fit(train_ds, val_ds)

print("Training complete. Checkpoint saved to outputs/minimal_run/checkpoints/")
