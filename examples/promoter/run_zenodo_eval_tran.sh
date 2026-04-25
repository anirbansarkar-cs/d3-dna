#!/usr/bin/env bash
#SBATCH --job-name=promo_tran_zenodo
#SBATCH --partition=kooq
#SBATCH --qos=koolab
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

source /grid/koo/home/duran/miniforge3/etc/profile.d/conda.sh
conda activate d3-dna

OUT_SAMPLES=generated_zenodo_tran
OUT_EVAL=eval_zenodo_tran
ORACLE_LOCAL=/grid/koo/home/shared/d3/oracle_weights/promoter/best.sei.model.pth.tar

echo "==[$(date)]== sample (transformer, bf16, 100 steps, paired_repeat=1, --use-test-labels)"
python sample.py \
    --config config_transformer.yaml \
    --use-test-labels \
    --steps 100 \
    --paired-repeat 1 \
    --output-dir "$OUT_SAMPLES"

# evaluate.py expects sample*.npz; copy / rename samples.npz -> sample.npz
if [[ -f "$OUT_SAMPLES/samples.npz" && ! -f "$OUT_SAMPLES/sample.npz" ]]; then
    cp "$OUT_SAMPLES/samples.npz" "$OUT_SAMPLES/sample.npz"
fi

echo "==[$(date)]== evaluate (mse, ks, js, auroc)"
python evaluate.py \
    --config config_transformer.yaml \
    --samples-dir "$OUT_SAMPLES" \
    --oracle-file "$ORACLE_LOCAL" \
    --paired-repeat 1 \
    --kmer-ks 6 \
    --output-dir "$OUT_EVAL"

echo "==[$(date)]== done"
