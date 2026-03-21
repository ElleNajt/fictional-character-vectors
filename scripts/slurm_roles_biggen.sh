#!/bin/bash
#SBATCH --job-name=roles_biggen
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=8:00:00

# Generate responses + extract activations for 275 Lu et al. roles
# on 90 BiGGen-Bench questions. Compares role eigenspectra with
# fictional character eigenspectra from the same questions.
#
# Usage:
#   cd /workspace-vast/lnajt
#   sbatch scripts/slurm_roles_biggen.sh

set -e

MODEL="Qwen/Qwen3-32B"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
ROLES_DIR="$REPO_ROOT/data/roles/role_instructions"
QUESTIONS_FILE="$REPO_ROOT/data/biggen_bench_questions.jsonl"
OUTPUT_DIR="$REPO_ROOT/outputs/roles_biggen"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/roles_biggen_${SLURM_JOB_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"

PYTHON="$REPO_ROOT/.venv/bin/python3"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source <(grep -v '^#' "$REPO_ROOT/.env" | sed 's/ #.*//')
    set +a
fi

[ -n "$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NUM_ROLES=$(ls -1 "$ROLES_DIR"/*.json 2>/dev/null | wc -l)
NUM_QUESTIONS=$(wc -l <"$QUESTIONS_FILE")
echo "=== Roles BiGGen-Bench Experiment ==="
echo "Model: $MODEL"
echo "Roles: $NUM_ROLES"
echo "Questions: $NUM_QUESTIONS"
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

# Phase 2: Extract activations
echo "=== Phase 2: Extracting activations ==="
$PYTHON pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations" \
    --layers 32 \
    --tensor_parallel_size 4

echo ""
echo "=== Done ==="
echo "Responses: $(ls -1 $OUTPUT_DIR/responses/*.jsonl | wc -l)"
echo "Activations: $(ls -1 $OUTPUT_DIR/activations/*.pt | wc -l)"
