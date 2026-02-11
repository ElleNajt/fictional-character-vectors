#!/bin/bash
#SBATCH --job-name=char_vectors
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

# Extract activations for fictional characters
#
# Usage:
#   cd fictional-character-vectors
#   python scripts/prepare_roles.py  # generate role files first
#
#   # Run for specific model:
#   MODEL=Qwen/Qwen3-32B sbatch scripts/slurm_extract.sh
#   MODEL=google/gemma-2-27b-it sbatch scripts/slurm_extract.sh
#
#   # Or use defaults (Qwen3-32B):
#   sbatch scripts/slurm_extract.sh

set -e

# Model can be set via environment variable, default to Qwen3-32B
MODEL="${MODEL:-Qwen/Qwen3-32B}"

# SLURM copies script to temp location, so use SLURM_SUBMIT_DIR
REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
ROLES_DIR="$REPO_ROOT/data/roles/instructions"

# Create model slug for output directory
MODEL_SLUG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')

# Timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$REPO_ROOT/outputs/${MODEL_SLUG}_$TIMESTAMP"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPO_ROOT/logs"

# Redirect output to logs directory
exec >"$REPO_ROOT/logs/char_vectors_${SLURM_JOB_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"

# Set up virtual environment if needed
if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$REPO_ROOT/.venv"
fi
source "$REPO_ROOT/.venv/bin/activate"

# Load .env if exists (for HF_TOKEN, etc.)
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

# HuggingFace token for gated models (Gemma, Llama, etc.)
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Install dependencies
pip install -e . --quiet

NUM_ROLES=$(ls -1 "$ROLES_DIR"/*.json 2>/dev/null | wc -l)
echo "=== Fictional Character Vector Extraction ==="
echo "Model: $MODEL"
echo "Roles: $NUM_ROLES characters"
echo "Output: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Phase 1: Generate responses
echo "=== Phase 1: Generating responses ==="
python pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$ASSISTANT_AXIS/data/extraction_questions.jsonl" \
    --output_dir "$OUTPUT_DIR/responses" \
    --question_count 240 \
    --tensor_parallel_size 4

echo ""

# Phase 2: Extract activations
echo "=== Phase 2: Extracting activations ==="
python pipeline/2_activations.py \
    --model "$MODEL" \
    --response_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations" \
    --layers 32 \
    --tensor_parallel_size 4

echo ""
echo "=== Done ==="
echo "Responses: $OUTPUT_DIR/responses"
echo "Activations: $OUTPUT_DIR/activations"
