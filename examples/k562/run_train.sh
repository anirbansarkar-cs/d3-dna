#!/bin/bash
#SBATCH --job-name=d3dna_k562_train
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Activate your conda environment (update for your system)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

cd "$(dirname "$0")"

python train.py "$@"
