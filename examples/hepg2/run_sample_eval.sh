#!/bin/bash
#SBATCH --job-name=d3dna_hepg2_eval
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Activate your conda environment (update for your system)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

cd "$(dirname "$0")"

echo "=== Sampling ==="
python sample.py --steps 20 --replicates 5

echo ""
echo "=== Evaluation (SP-MSE, built-in) ==="
python evaluate.py \
    --data data/lenti_MPRA_HepG2_data.h5 \
    --oracle data/oracle_best_model.ckpt

# For full 5-metric evaluation, add --eval-pipeline:
# python evaluate.py \
#     --data data/lenti_MPRA_HepG2_data.h5 \
#     --oracle data/oracle_best_model.ckpt \
#     --eval-pipeline /path/to/d3_evaluation_pipeline

echo ""
echo "=== Done ==="
