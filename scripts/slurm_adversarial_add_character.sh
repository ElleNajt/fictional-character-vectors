#!/bin/bash
#SBATCH --job-name=adv_add_char
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

REPO_ROOT="/workspace-vast/lnajt"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/adv_add_char_${SLURM_JOB_ID}.out" 2>&1

source "$REPO_ROOT/.venv/bin/activate"

echo "=== Add Adversarial Character ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

python3 "$REPO_ROOT/scripts/adversarial_add_character.py"

echo ""
echo "=== Now re-running token projections ==="
python3 "$REPO_ROOT/scripts/adversarial_token_projections.py"

echo "=== Done ==="
