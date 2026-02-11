#!/bin/bash
#
# Run the full extraction pipeline: prepare roles, generate responses, extract activations
#
# Usage:
#   ./scripts/run_extraction.sh [--model MODEL] [--output-dir DIR] [--gpus N]
#
# Example:
#   ./scripts/run_extraction.sh --model Qwen/Qwen2.5-32B-Instruct --gpus 4
#
# Requirements:
#   - GPU machine with sufficient VRAM (4x80GB for 32B model with TP=4)
#   - Python environment with vLLM installed
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"

# Defaults
MODEL="Qwen/Qwen2.5-32B-Instruct"
OUTPUT_DIR="$REPO_ROOT/outputs"
TENSOR_PARALLEL_SIZE=""  # auto-detect
QUESTION_COUNT=240

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --questions)
            QUESTION_COUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Derive model slug for output paths
MODEL_SLUG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')

echo "=== Fictional Character Vector Extraction Pipeline ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Prepare role files
echo "=== Step 1: Preparing role files ==="
python "$SCRIPT_DIR/prepare_roles.py"
ROLES_DIR="$REPO_ROOT/data/roles/instructions"
echo ""

# Step 2: Generate responses using assistant-axis pipeline
echo "=== Step 2: Generating responses ==="
RESPONSES_DIR="$OUTPUT_DIR/$MODEL_SLUG/responses"
mkdir -p "$RESPONSES_DIR"

cd "$ASSISTANT_AXIS"

TP_ARG=""
if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
    TP_ARG="--tensor_parallel_size $TENSOR_PARALLEL_SIZE"
fi

python pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$ASSISTANT_AXIS/data/extraction_questions.jsonl" \
    --output_dir "$RESPONSES_DIR" \
    --question_count "$QUESTION_COUNT" \
    $TP_ARG

echo ""

# Step 3: Extract activations
echo "=== Step 3: Extracting activations ==="
ACTIVATIONS_DIR="$OUTPUT_DIR/$MODEL_SLUG/activations"
mkdir -p "$ACTIVATIONS_DIR"

python pipeline/2_activations.py \
    --model "$MODEL" \
    --response_dir "$RESPONSES_DIR" \
    --output_dir "$ACTIVATIONS_DIR" \
    --layers 32 \
    $TP_ARG

echo ""
echo "=== Done ==="
echo "Responses: $RESPONSES_DIR"
echo "Activations: $ACTIVATIONS_DIR"
