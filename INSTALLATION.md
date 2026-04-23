# d3-dna Installation Guide

Tested on: RHEL 8 (GLIBC 2.28), SLURM cluster with V100/H100 GPUs, conda (miniforge3), 2026-04-09.

## 1. Create a conda environment

```bash
conda create -n d3-dna python=3.11 -y
conda activate d3-dna
```

Python 3.11 is recommended. The package requires >=3.9.

## 2. Install core package

The README says `pip install d3-dna`, but the package is **not yet published on PyPI**. Install from the local source instead:

```bash
cd /path/to/d3-dna
pip install -e .
```

This installs: torch, pytorch-lightning, omegaconf, numpy, h5py, tqdm, einops.

### Issue: PyTorch CUDA version too new for cluster GPUs

By default, `pip install -e .` pulls the latest PyTorch (e.g., 2.11.0 with CUDA 13.0). This causes two problems on older clusters:

1. **NVIDIA driver too old** -- CUDA 13.0 requires a newer driver than what many clusters have (e.g., driver 12040 / CUDA 12.4).
2. **Compute capability dropped** -- PyTorch 2.6+ dropped support for V100 GPUs (compute capability 7.0).

**Fix:** Pin PyTorch to a version compatible with your cluster's CUDA driver and GPU hardware:

```bash
# For clusters with CUDA 12.x drivers and V100/A100/H100 GPUs:
pip install 'torch>=2.0.0,<2.6.0'
```

This installs PyTorch 2.5.1 with CUDA 12.4, which supports V100 (CC 7.0) through H100 (CC 9.0).

## 3. Install optional extras

### Weights & Biases logging

```bash
pip install -e ".[logging]"
# or standalone:
pip install wandb
```

No issues encountered.

### Flash attention

The README says `pip install d3-dna[flash]`, but flash-attn requires building from source on most systems. Multiple issues arise:

#### Issue 1: `ModuleNotFoundError: No module named 'torch'`

flash-attn's `setup.py` needs torch at build time, but pip's build isolation creates a clean venv without it.

**Fix:** Use `--no-build-isolation`:

```bash
pip install flash-attn --no-build-isolation
```

#### Issue 2: `CUDA_HOME environment variable is not set`

flash-attn needs to compile CUDA kernels and requires `nvcc`.

**Fix:** Set `CUDA_HOME` to your cluster's CUDA toolkit:

```bash
export CUDA_HOME=/cm/shared/apps/cuda12.3/toolkit/current
export PATH=$CUDA_HOME/bin:$PATH
```

Adjust the path to match your cluster's CUDA installation. Use `module avail cuda` or `ls /usr/local/cuda*` to find it.

#### Issue 3: `ModuleNotFoundError: No module named 'psutil'`

**Fix:** Install build dependencies first:

```bash
pip install psutil ninja
```

#### Issue 4: `Invalid cross-device link` (NFS home directories)

pip's wheel cache on NFS fails to create hard links across filesystems.

**Fix:** Disable the pip cache:

```bash
pip install flash-attn --no-build-isolation --no-cache-dir
```

#### Issue 5: `GLIBC_2.32 not found` (RHEL 8 / CentOS 8)

flash-attn >= 2.6.0 is compiled against GLIBC 2.32+. RHEL 8 ships GLIBC 2.28.

**Fix:** Pin an older flash-attn version:

```bash
pip install "flash-attn==2.5.8" --no-build-isolation --no-cache-dir
```

#### Complete flash-attn install command

Combining all fixes (run on a GPU node via `srun`):

```bash
srun --partition=gpuq --gres=gpu:1 --time=00:15:00 bash -c '
  conda activate d3-dna
  pip install psutil ninja
  export CUDA_HOME=/cm/shared/apps/cuda12.3/toolkit/current
  export PATH=$CUDA_HOME/bin:$PATH
  pip install "flash-attn==2.5.8" --no-build-isolation --no-cache-dir
'
```

Build takes ~5-10 minutes. Must be run on a GPU node (needs `nvcc` and CUDA headers).

## 4. Verify installation

### On login node (CPU-only, no GPU expected):

```bash
conda activate d3-dna
python -c "from d3_dna import D3Trainer, D3Sampler, D3Evaluator; from d3_dna.models import TransformerModel, ConvolutionalModel; print('OK')"
```

You will see CUDA warnings (Can't initialize NVML) -- this is normal on a login node without GPUs.

### On GPU node:

```bash
srun --partition=gpuq --gres=gpu:1 --time=00:02:00 bash -c '
  conda activate d3-dna
  python -c "
import torch
print(\"CUDA:\", torch.cuda.is_available())
print(\"GPU:\", torch.cuda.get_device_name(0))
from d3_dna import D3Trainer, D3Sampler
print(\"d3-dna OK\")
try:
    import flash_attn
    print(\"flash-attn\", flash_attn.__version__)
except ImportError:
    print(\"flash-attn not installed (optional)\")
"
'
```

## Working package versions

These versions were confirmed working together on this cluster:

| Package | Version |
|---|---|
| python | 3.11.15 |
| torch | 2.5.1 (CUDA 12.4) |
| pytorch-lightning | 2.6.1 |
| flash-attn | 2.5.8 |
| wandb | 0.25.1 |
| numpy | 2.4.4 |
| h5py | 3.16.0 |
| omegaconf | 2.3.0 |
| einops | 0.8.2 |
| tqdm | 4.67.3 |
