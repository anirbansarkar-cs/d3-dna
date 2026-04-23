#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_tran_only
#SBATCH --output=examples/promoter/eval_only_tran_%j.log
#SBATCH --chdir=/grid/koo/home/duran/d3-dna

source /grid/koo/home/duran/miniforge3/etc/profile.d/conda.sh
conda activate d3-dna

nvidia-smi | head -20

OUTDIR=examples/promoter/final_eval/tran_epoch50_full_hybrid

python -u examples/promoter/evaluate.py \
  --samples "$OUTDIR/samples.npz" \
  --data /grid/koo/home/shared/d3/data/promoter/Promoter_data.npz \
  --paired-repeat 5 \
  --tests mse,ks,js,auroc \
  --kmer-ks 1-7 \
  --output-dir "$OUTDIR"
