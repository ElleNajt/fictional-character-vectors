#!/bin/bash
#SBATCH --job-name=degen_q
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

# Generate responses + extract activations for degenerate questions
# ("Huh?", ".", "?", "hi", "ok", "what", "hmm", "tell me something")
#
# Usage:
#   cd fictional-character-vectors
#   sbatch scripts/slurm_degenerate_questions.sh

set -e

MODEL="Qwen/Qwen3-32B"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
ROLES_DIR="$REPO_ROOT/data/roles/instructions"
QUESTIONS_FILE="$REPO_ROOT/data/degenerate_questions.jsonl"
OUTPUT_DIR="$REPO_ROOT/outputs/degenerate_questions"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/degen_q_${SLURM_JOB_ID}.out" 2>&1

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

NUM_ROLES=$(ls -1 "$ROLES_DIR"/*.json 2>/dev/null | wc -l)
NUM_QUESTIONS=$(wc -l <"$QUESTIONS_FILE")
echo "=== Degenerate Question Experiment ==="
echo "Model: $MODEL"
echo "Roles: $NUM_ROLES characters"
echo "Questions: $NUM_QUESTIONS degenerate questions"
echo "Output: $OUTPUT_DIR"
echo ""

# Phase 1: Generate responses
echo "=== Phase 1: Generating responses ==="
python pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT_DIR/responses" \
    --question_count "$NUM_QUESTIONS" \
    --tensor_parallel_size 4

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
echo "=== Done ==="
echo "Responses: $OUTPUT_DIR/responses"
echo "Activations: $OUTPUT_DIR/activations"
