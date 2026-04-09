"""Find the maximum batch size and optimal throughput for sampling on this GPU."""
import torch
import time
from omegaconf import OmegaConf
from d3_dna import D3Sampler
from d3_dna.models import TransformerModel

cfg = OmegaConf.load("config.yaml")
model = TransformerModel(cfg)
sampler = D3Sampler(cfg)
sampler.load(
    checkpoint="/grid/koo/home/shared/d3/trained_weights/promoter_09242025/model-epoch=175-val_loss=1119.9065.ckpt",
    model=model,
    device="cuda",
)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

# Warmup with small batch
labels = torch.randn(8, 1024, 1)
_ = sampler.generate_batched(num_samples=8, labels=labels, batch_size=8, steps=5)
print("Warmup done\n")

results = []
for bs in [64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    labels = torch.randn(bs, 1024, 1)
    try:
        t0 = time.time()
        seqs = sampler.generate_batched(num_samples=bs, labels=labels, batch_size=bs, steps=128)
        elapsed = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        throughput = bs / elapsed
        results.append((bs, peak_gb, elapsed, throughput))
        print(f"  bs={bs:>5d}: {peak_gb:5.1f} GB | {elapsed:6.1f}s | {throughput:5.1f} seq/s")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err = "OOM" if "out of memory" in str(e).lower() else str(e)[:60]
        print(f"  bs={bs:>5d}: FAIL ({err})")
        break

if results:
    best = max(results, key=lambda r: r[3])
    print(f"\nOptimal: batch_size={best[0]} ({best[3]:.1f} seq/s, {best[1]:.1f} GB)")
