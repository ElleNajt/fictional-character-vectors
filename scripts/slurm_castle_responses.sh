#!/bin/bash
#SBATCH --job-name=castle_resp
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=6:00:00

# Generate security reviews for CASTLE benchmark + extract per-token
# activations at layer 32. 250 programs × (generate + forward pass).
#
# Usage:
#   cd fictional-character-vectors
#   sbatch scripts/slurm_castle_responses.sh

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON="$REPO_ROOT/.venv/bin/python3"

mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/castle_responses_${SLURM_JOB_ID}.out" 2>&1

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source <(grep -v '^#' "$REPO_ROOT/.env" | sed 's/ #.*//')
    set +a
fi

if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== CASTLE Response Activations ==="
echo "Python: $PYTHON"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

$PYTHON "$REPO_ROOT/scripts/castle_response_activations.py"

echo ""
echo "=== Done ==="
echo "Results: $REPO_ROOT/outputs/castle_response_activations/"
