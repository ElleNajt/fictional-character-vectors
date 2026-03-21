#!/bin/bash
#SBATCH --job-name=fix_aa_clamp
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

REPO_ROOT="/workspace-vast/lnajt"
PYTHON="$REPO_ROOT/.venv/bin/python3"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/fix_aa_clamping_${SLURM_JOB_ID}.out" 2>&1

echo "=== Fix AA clamping (layers_46:54-p0.25) ==="
echo "Node: $(hostname)"

cd "$REPO_ROOT"

# Step 1: Re-generate aa_clamped with correct experiment
$PYTHON scripts/fix_aa_clamping.py --resume

# Step 2: Re-run token projections
$PYTHON scripts/adversarial_token_projections.py

echo "=== Done ==="
