#!/bin/bash
#SBATCH --job-name=llama_chars
#SBATCH --partition=overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

# Extract activations for fictional characters using Llama 3.3 70B
# Includes LLM judge step for filtering to fully role-playing responses

set -e

MODEL="meta-llama/Llama-3.3-70B-Instruct"
TARGET_LAYER=40 # Llama 3.3 70B target layer from Christina's paper

REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
ROLES_DIR="$REPO_ROOT/data/roles/instructions"

MODEL_SLUG="llama-3.3-70b"
# Resume existing run
OUTPUT_DIR="$REPO_ROOT/outputs/llama-3.3-70b_20260212_155042"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPO_ROOT/logs"

exec >"$REPO_ROOT/logs/llama_${SLURM_JOB_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"

# Set up virtual environment
if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$REPO_ROOT/.venv"
fi
source "$REPO_ROOT/.venv/bin/activate"

# Load .env for HF_TOKEN and OPENAI_API_KEY
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

pip install -e . --quiet

NUM_ROLES=$(ls -1 "$ROLES_DIR"/*.json 2>/dev/null | wc -l)
echo "=== Llama 3.3 70B Character Vector Extraction ==="
echo "Model: $MODEL"
echo "Target Layer: $TARGET_LAYER"
echo "Roles: $NUM_ROLES characters"
echo "Output: $OUTPUT_DIR"
echo ""

# Phase 1: Generate responses
echo "=== Phase 1: Generating responses ==="
python pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$ASSISTANT_AXIS/data/extraction_questions.jsonl" \
    --output_dir "$OUTPUT_DIR/responses" \
    --question_count 240 \
    --tensor_parallel_size 8

echo ""

# Phase 2: Extract activations (can run in parallel with Phase 3)
echo "=== Phase 2: Extracting activations ==="
python pipeline/2_activations.py \
    --model "$MODEL" \
    --response_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations" \
    --layers $TARGET_LAYER \
    --tensor_parallel_size 8 &

ACTIVATIONS_PID=$!

# Phase 3: Judge responses (runs in parallel with Phase 2)
echo "=== Phase 3: Judging responses ==="
if [ -n "$OPENAI_API_KEY" ]; then
    python pipeline/3_judge.py \
        --responses_dir "$OUTPUT_DIR/responses" \
        --output_dir "$OUTPUT_DIR/scores"
else
    echo "WARNING: OPENAI_API_KEY not set, skipping judge step"
    echo "You can run this later with: python pipeline/3_judge.py --responses_dir $OUTPUT_DIR/responses --output_dir $OUTPUT_DIR/scores"
fi

# Wait for activations to finish
wait $ACTIVATIONS_PID

echo ""

# Phase 4: Compute per-role vectors (only score=3)
echo "=== Phase 4: Computing filtered vectors ==="
if [ -d "$OUTPUT_DIR/scores" ]; then
    python pipeline/4_vectors.py \
        --activations_dir "$OUTPUT_DIR/activations" \
        --scores_dir "$OUTPUT_DIR/scores" \
        --output_dir "$OUTPUT_DIR/vectors"
else
    echo "Skipping vector filtering (no scores directory)"
fi

echo ""
echo "=== Done ==="
echo "Responses: $OUTPUT_DIR/responses"
echo "Activations: $OUTPUT_DIR/activations"
echo "Scores: $OUTPUT_DIR/scores"
echo "Vectors: $OUTPUT_DIR/vectors"
