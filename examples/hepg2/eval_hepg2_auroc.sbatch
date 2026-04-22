#!/bin/bash
#SBATCH --job-name=d3_auroc_hepg2
#SBATCH --output=/grid/koo/home/asarkar/d3-dna/examples/hepg2/slurm_auroc_hepg2_%j.out
#SBATCH --error=/grid/koo/home/asarkar/d3-dna/examples/hepg2/slurm_auroc_hepg2_%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=h100

set -euo pipefail

WORKDIR=/grid/koo/home/asarkar/d3-dna/examples/hepg2
DATA=/grid/koo/home/asarkar/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/lenti_MPRA_HepG2_data.h5
GEN_DIR=${WORKDIR}/generated_package_100steps
EVAL_DIR=${WORKDIR}/eval_results_package_api_auroc

mkdir -p "${EVAL_DIR}"
source /grid/it/data/elzar/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate d3_cuda118
cd "${WORKDIR}"

echo "=== HepG2 AUROC (70/10/20 split, held-out test) ==="
python run_package_eval_full.py \
    --samples-dir "${GEN_DIR}" \
    --data "${DATA}" \
    --oracle /dev/null \
    --tests auroc \
    --output-dir "${EVAL_DIR}"

echo "=== Done ==="
