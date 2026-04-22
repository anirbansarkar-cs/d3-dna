#!/bin/bash
#SBATCH --job-name=d3_eval_hepg2
#SBATCH --output=/grid/koo/home/asarkar/d3-dna/examples/hepg2/slurm_eval_hepg2_%j.out
#SBATCH --error=/grid/koo/home/asarkar/d3-dna/examples/hepg2/slurm_eval_hepg2_%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=h100

set -euo pipefail

WORKDIR=/grid/koo/home/asarkar/d3-dna/examples/hepg2
DATA=/grid/koo/home/asarkar/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/lenti_MPRA_HepG2_data.h5
ORACLE=/grid/koo/home/shared/d3/oracle_weights/lentimpra/HepG2/best_model-epoch=15-val_pearson=0.741.ckpt
GEN_DIR=${WORKDIR}/generated_package_100steps
EVAL_DIR=${WORKDIR}/eval_results_package_100steps

mkdir -p "${EVAL_DIR}"
source /grid/it/data/elzar/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate d3_cuda118
cd "${WORKDIR}"

echo "=== HepG2 eval (100-step, 5-metric suite) ==="
python evaluate.py \
    --samples-dir "${GEN_DIR}" \
    --data "${DATA}" \
    --oracle "${ORACLE}" \
    --eval-pipeline /grid/koo/home/asarkar/d3_evaluation_pipeline \
    --output-dir "${EVAL_DIR}"

echo "=== Eval done ==="
