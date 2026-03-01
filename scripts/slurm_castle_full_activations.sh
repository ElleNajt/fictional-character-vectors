#!/bin/bash
#SBATCH --job-name=castle_full_acts
#SBATCH --output=logs/castle_full_acts_%j.out
#SBATCH --error=logs/castle_full_acts_%j.out
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --time=6:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CASTLE Full Activation Extraction ==="
echo "Python: $REPO_ROOT/.venv/bin/python3"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo ""

mkdir -p logs outputs/castle_full_activations/activations

exec "$REPO_ROOT/.venv/bin/python3" scripts/castle_full_activations.py
