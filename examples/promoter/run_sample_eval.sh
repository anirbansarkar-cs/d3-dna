#!/bin/bash
#SBATCH --job-name=d3dna_promoter_eval
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

CHECKPOINT="${CHECKPOINT:-outputs/promoter_transformer/checkpoints/last.ckpt}"
CONFIG="${CONFIG:-config_transformer.yaml}"

echo "=== Sampling (checkpoint=$CHECKPOINT) ==="
python sample.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --use-test-labels \
    --steps 128 \
    --batch-size 64

echo ""
echo "=== Evaluation (MSE, KS, JS, AUROC) ==="
python evaluate.py --samples generated/samples.npz

echo ""
echo "=== Done ==="
