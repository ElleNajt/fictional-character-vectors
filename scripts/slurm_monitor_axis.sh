#!/bin/bash
#SBATCH --job-name=monitor_axis
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=6:00:00

# Generate responses + extract activations for monitor roles
# (monitor and paranoid_monitor only — 2 roles × 5 prompts × 240 questions)
#
# This is a reproducibility test: can we extend the pipeline with a new persona?
#
# Usage:
#   cd fictional-character-vectors
#   sbatch scripts/slurm_monitor_axis.sh

set -e

MODEL="Qwen/Qwen3-32B"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
ROLES_DIR="$REPO_ROOT/data/roles/instructions"
QUESTIONS_FILE="$ASSISTANT_AXIS/data/extraction_questions.jsonl"
OUTPUT_DIR="$REPO_ROOT/outputs/monitor_axis"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/monitor_axis_${SLURM_JOB_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"

source "$REPO_ROOT/.venv/bin/activate"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source <(grep -v '^#' "$REPO_ROOT/.env" | sed 's/ #.*//')
    set +a
fi

if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

pip install -e . --quiet

echo "=== Monitor Axis Experiment ==="
echo "Model: $MODEL"
echo "Roles: monitor, paranoid_monitor"
echo "Questions: 240 (standard extraction battery)"
echo "Output: $OUTPUT_DIR"
echo ""

# Phase 1: Generate responses (monitor roles only)
echo "=== Phase 1: Generating responses ==="
python pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT_DIR/responses" \
    --question_count 240 \
    --tensor_parallel_size 4 \
    --roles monitor paranoid_monitor

echo ""

# Phase 2: Extract activations (layer 32 only)
echo "=== Phase 2: Extracting activations ==="
python pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations" \
    --layers 32 \
    --tensor_parallel_size 4

echo ""

# Note: we skip pipeline steps 3 (judge) and 4 (vectors) here.
# 4_vectors.py hardcodes "default" detection for unscored roles.
# Instead, compute_monitor_axis.py averages activations directly.

echo "=== Done ==="
echo "Responses: $OUTPUT_DIR/responses"
echo "Activations: $OUTPUT_DIR/activations"
echo ""
echo "Next: run 'python scripts/compute_monitor_axis.py' to compute axis vectors"
