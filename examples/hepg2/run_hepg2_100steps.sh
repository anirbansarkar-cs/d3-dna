#!/bin/bash
# 5 parallel sampling replicates (100 steps) + dependent eval job (5-metric suite).

WORKDIR="/grid/koo/home/asarkar/d3-dna/examples/hepg2"
CKPT="/grid/koo/home/shared/d3/trained_weights/lentimpra/d3-tran-Hepg2/model-epoch=259-val_loss=250.9961.ckpt"
DATA="/grid/koo/home/asarkar/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/lenti_MPRA_HepG2_data.h5"
ORACLE="/grid/koo/home/asarkar/D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/oracle_models/best_model_hepg2-epoch=15-val_pearson=0.705.ckpt"
GEN_DIR="${WORKDIR}/generated_package_100steps"
EVAL_DIR="${WORKDIR}/eval_results_package_100steps"
STEPS=100

mkdir -p "${GEN_DIR}" "${EVAL_DIR}"

SAMPLE_JOBID=$(sbatch --parsable <<SBATCH
#!/bin/bash
#SBATCH --job-name=d3_sample_hepg2
#SBATCH --output=${WORKDIR}/slurm_sample_hepg2_%a.out
#SBATCH --error=${WORKDIR}/slurm_sample_hepg2_%a.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=h100
#SBATCH --qos=default
#SBATCH --array=0-4

source /grid/it/data/elzar/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate d3_cuda118

cd "${WORKDIR}"

REP=\${SLURM_ARRAY_TASK_ID}
echo "=== HepG2 sampling replicate \${REP} with ${STEPS} steps ==="
echo "Checkpoint: ${CKPT}"

python sample.py \\
    --checkpoint "${CKPT}" \\
    --steps ${STEPS} \\
    --replicates 1 \\
    --output-dir "${GEN_DIR}" \\
    --rep-offset \${REP}

echo "=== Replicate \${REP} done ==="
SBATCH
)

echo "Submitted sampling array job: ${SAMPLE_JOBID} (5 parallel tasks)"

EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${SAMPLE_JOBID} <<SBATCH
#!/bin/bash
#SBATCH --job-name=d3_eval_hepg2
#SBATCH --output=${WORKDIR}/slurm_eval_hepg2.out
#SBATCH --error=${WORKDIR}/slurm_eval_hepg2.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=h100
#SBATCH --qos=default

source /grid/it/data/elzar/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate d3_cuda118

cd "${WORKDIR}"

echo "=== HepG2 evaluation (100-step, 5-metric suite) ==="
python evaluate.py \\
    --samples-dir "${GEN_DIR}" \\
    --data "${DATA}" \\
    --oracle "${ORACLE}" \\
    --eval-pipeline /grid/koo/home/asarkar/d3_evaluation_pipeline \\
    --output-dir "${EVAL_DIR}"

echo "=== Evaluation done ==="
SBATCH
)

echo "Submitted eval job: ${EVAL_JOBID} (depends on ${SAMPLE_JOBID})"
echo "Monitor: squeue -u \$(whoami) | grep d3_"
