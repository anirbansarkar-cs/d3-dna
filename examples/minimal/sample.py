"""Sample sequences from a trained checkpoint."""
import glob
import torch
from omegaconf import OmegaConf
from d3_dna import D3Sampler
from d3_dna.models import TransformerModel

cfg = OmegaConf.load('config.yaml')

checkpoints = glob.glob('outputs/minimal_run/checkpoints/*.ckpt')
assert checkpoints, "Run train.py first to generate a checkpoint."
checkpoint_path = sorted(checkpoints)[-1]

model = TransformerModel(cfg)

sampler = D3Sampler(cfg)
sampler.load(checkpoint=checkpoint_path, model=model,
             device='cuda' if torch.cuda.is_available() else 'cpu')

labels = torch.rand(10, 1)  # 10 sequences, 1D global labels

sequences = sampler.generate_batched(
    num_samples=10,
    labels=labels,
    batch_size=10,
)

print(f"Generated {sequences.shape[0]} sequences of length {sequences.shape[1]}")
sampler.save(sequences, 'generated.fasta', format='fasta')
print("Saved to generated.fasta")
