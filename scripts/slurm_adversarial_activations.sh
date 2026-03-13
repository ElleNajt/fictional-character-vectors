#!/bin/bash
#SBATCH --job-name=adv_acts
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

REPO_ROOT="/workspace-vast/lnajt"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/adv_acts_${SLURM_JOB_ID}.out" 2>&1

source "$REPO_ROOT/.venv/bin/activate"

echo "=== Adversarial Activations ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

python3 "$REPO_ROOT/scripts/adversarial_activations.py"

echo "=== Done ==="
