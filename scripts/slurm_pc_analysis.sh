#!/bin/bash
#SBATCH --job-name=pc_analysis
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Compute PC analysis for all universes

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR}"

cd "$REPO_ROOT"
source ".venv/bin/activate"

echo "=== PC Analysis Computation ==="
echo "Start: $(date)"
echo ""

python scripts/compute_pc_analysis.py \
    --output results/pc_analysis.pkl \
    --model qwen3-32b \
    --layer 32 \
    --n-pcs 5 \
    --n-top-questions 10 \
    --skip-all \
    --skip-correlations

echo ""
echo "=== Done ==="
echo "End: $(date)"
