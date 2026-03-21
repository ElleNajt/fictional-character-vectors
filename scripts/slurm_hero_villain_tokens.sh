#!/bin/bash
#SBATCH --job-name=hv_tokens
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

REPO_ROOT="/workspace-vast/lnajt"
PYTHON="$REPO_ROOT/.venv/bin/python3"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/hv_tokens_${SLURM_JOB_ID}.out" 2>&1

echo "=== Hero/villain token projections ==="
echo "Node: $(hostname)"

cd "$REPO_ROOT"
$PYTHON scripts/hero_villain_token_projections.py --resume
