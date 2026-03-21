#!/bin/bash
#SBATCH --job-name=biggen_q
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00

# Generate responses + extract activations for BiGGen-Bench questions
# 90 diverse task questions (10 per category) to test whether
# low dimensionality is a property of the psychographic question battery
#
# Usage:
#   cd /workspace-vast/lnajt
#   sbatch scripts/slurm_biggen_bench.sh

set -e

MODEL="Qwen/Qwen3-32B"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
ROLES_DIR="$REPO_ROOT/data/roles/instructions"
QUESTIONS_FILE="$REPO_ROOT/data/biggen_bench_questions.jsonl"
OUTPUT_DIR="$REPO_ROOT/outputs/biggen_bench"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/biggen_q_${SLURM_JOB_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"

PYTHON="$REPO_ROOT/.venv/bin/python3"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source <(grep -v '^#' "$REPO_ROOT/.env" | sed 's/ #.*//')
    set +a
fi

if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

NUM_ROLES=$(ls -1 "$ROLES_DIR"/*.json 2>/dev/null | wc -l)
NUM_QUESTIONS=$(wc -l <"$QUESTIONS_FILE")
echo "=== BiGGen-Bench Question Experiment ==="
echo "Python: $PYTHON"
echo "Model: $MODEL"
echo "Roles: $NUM_ROLES characters"
echo "Questions: $NUM_QUESTIONS diverse task questions"
echo "Output: $OUTPUT_DIR"
echo ""

# Phase 1: Generate responses
echo "=== Phase 1: Generating responses ==="
$PYTHON pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT_DIR/responses" \
    --question_count "$NUM_QUESTIONS" \
    --tensor_parallel_size 4

echo ""

# Phase 2: Extract activations (layer 32 only)
echo "=== Phase 2: Extracting activations ==="
$PYTHON pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations" \
    --layers 32 \
    --tensor_parallel_size 4

echo ""
echo "=== Done ==="
echo "Responses: $OUTPUT_DIR/responses"
echo "Activations: $OUTPUT_DIR/activations"
