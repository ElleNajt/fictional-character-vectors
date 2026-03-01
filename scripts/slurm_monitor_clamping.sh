#!/bin/bash
#SBATCH --job-name=monitor_act
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2:00:00

# Monitor axis activation experiment: extract per-token projections onto
# the monitor axis for CASTLE benchmark code snippets.
#
# 250 forward passes (no generation), ~30 min expected.
#
# Usage:
#   cd fictional-character-vectors
#   sbatch scripts/slurm_monitor_clamping.sh

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON="$REPO_ROOT/.venv/bin/python3"

mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/monitor_activations_${SLURM_JOB_ID}.out" 2>&1

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source <(grep -v '^#' "$REPO_ROOT/.env" | sed 's/ #.*//')
    set +a
fi

if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== Monitor Activation Experiment ==="
echo "Python: $PYTHON"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

$PYTHON "$REPO_ROOT/scripts/monitor_clamping_experiment.py"

echo ""
echo "=== Done ==="
echo "Results: $REPO_ROOT/outputs/monitor_activations/"
