#!/bin/bash
#SBATCH --job-name=d3dna_promoter_train
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Activate your conda environment (update for your system)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

cd "$(dirname "$0")"

# Defaults to config_transformer.yaml. To train the conv architecture:
#   sbatch run_train.sh --config config_conv.yaml --work-dir outputs/promoter_conv
python train.py "$@"
