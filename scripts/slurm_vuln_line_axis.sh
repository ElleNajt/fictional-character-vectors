#!/bin/bash
#SBATCH --job-name=vuln_line_axis
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON="$REPO_ROOT/.venv/bin/python3"

mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/vuln_line_axis_${SLURM_JOB_ID}.out" 2>&1

echo "=== Vuln Line Axis Computation ==="
echo "Node: $(hostname)"
echo ""

$PYTHON "$REPO_ROOT/scripts/compute_vuln_line_axis.py"

echo ""
echo "=== Done ==="
