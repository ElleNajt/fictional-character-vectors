#!/bin/bash
#SBATCH --job-name=castle_full_acts
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=6:00:00

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON="$REPO_ROOT/.venv/bin/python3"

mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/castle_full_acts_${SLURM_JOB_ID}.out" 2>&1

echo "=== CASTLE Full Activation Extraction ==="
echo "Python: $PYTHON"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

mkdir -p "$REPO_ROOT/outputs/castle_full_activations/activations"

$PYTHON "$REPO_ROOT/scripts/castle_full_activations.py"

echo ""
echo "=== Done ==="
