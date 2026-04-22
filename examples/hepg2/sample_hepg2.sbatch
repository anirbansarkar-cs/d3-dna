#!/bin/bash
#SBATCH --job-name=d3_sample_hepg2
#SBATCH --output=/grid/koo/home/asarkar/d3-dna/examples/hepg2/slurm_sample_hepg2_%x_%j.out
#SBATCH --error=/grid/koo/home/asarkar/d3-dna/examples/hepg2/slurm_sample_hepg2_%x_%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=h100

set -euo pipefail

: "${REP:?REP env var must be set (0-4)}"
WORKDIR=/grid/koo/home/asarkar/d3-dna/examples/hepg2
CKPT=/grid/koo/home/shared/d3/trained_weights/lentimpra/d3-tran-Hepg2/model-epoch=259-val_loss=250.9961.ckpt
GEN_DIR=${WORKDIR}/generated_package_100steps
STEPS=100

mkdir -p "${GEN_DIR}"
source /grid/it/data/elzar/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate d3_cuda118
cd "${WORKDIR}"

echo "=== HepG2 sampling REP=${REP} STEPS=${STEPS} ==="
echo "Checkpoint: ${CKPT}"

python sample.py \
    --checkpoint "${CKPT}" \
    --steps "${STEPS}" \
    --replicates 1 \
    --output-dir "${GEN_DIR}" \
    --rep-offset "${REP}"

echo "=== Replicate ${REP} done ==="
