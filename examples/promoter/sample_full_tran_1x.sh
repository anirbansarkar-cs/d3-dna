#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --job-name=tran_full_1x
#SBATCH --output=examples/promoter/sample_full_tran_1x_%j.log
#SBATCH --chdir=/grid/koo/home/duran/d3-dna

source /grid/koo/home/duran/miniforge3/etc/profile.d/conda.sh
conda activate d3-dna

nvidia-smi | head -20

OUTDIR=examples/promoter/final_eval/tran_epoch50_full_hybrid_1x
CKPT=/grid/koo/home/shared/d3/trained_weights/promoter/tran/checkpoint_50.pth

echo "=== Sampling (full test set, 1-per-TSS, hybrid shim) ==="
python -u examples/promoter/sample.py \
  --checkpoint "$CKPT" \
  --config examples/promoter/config_transformer.yaml \
  --hybrid-shim \
  --use-test-labels \
  --batch-size 64 \
  --steps 128 \
  --output-dir "$OUTDIR"

echo "=== Evaluation ==="
python -u examples/promoter/evaluate.py \
  --samples "$OUTDIR/samples.npz" \
  --data /grid/koo/home/shared/d3/data/promoter/Promoter_data.npz \
  --tests mse,ks,js,auroc \
  --kmer-ks 1-7 \
  --output-dir "$OUTDIR"
