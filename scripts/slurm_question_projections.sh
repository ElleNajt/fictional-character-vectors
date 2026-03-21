#!/bin/bash
#SBATCH --job-name=q_proj
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -e
cd "${SLURM_SUBMIT_DIR}"
source ".venv/bin/activate"

echo "=== Question Projections ==="
echo "Start: $(date)"

python scripts/compute_question_projections.py \
    --output results/question_projections.pkl \
    --layer 32 \
    --n-pcs 5

echo "Done: $(date)"
