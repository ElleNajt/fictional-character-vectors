#!/bin/bash
#SBATCH --job-name=new_acts
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

# Extract activations for NEW mythology/Shakespeare characters
# Run this after slurm_generate_new.sh completes

set -e

MODEL="Qwen/Qwen3-32B"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
OUTPUT_DIR="$REPO_ROOT/outputs/qwen3-32b_20260211_002840"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/new_acts_${SLURM_JOB_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"
source "$REPO_ROOT/.venv/bin/activate"

NUM_RESPONSES=$(ls -1 "$OUTPUT_DIR/responses"/*.jsonl 2>/dev/null | wc -l)
NUM_ACTIVATIONS=$(ls -1 "$OUTPUT_DIR/activations"/*.pt 2>/dev/null | wc -l)
NUM_NEW=$((NUM_RESPONSES - NUM_ACTIVATIONS))

echo "=== Extract Activations for New Characters ==="
echo "Model: $MODEL"
echo "Total responses: $NUM_RESPONSES"
echo "Existing activations: $NUM_ACTIVATIONS"
echo "New to extract: $NUM_NEW"
echo "Output: $OUTPUT_DIR/activations"
echo ""

python pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations" \
    --batch_size 16 \
    --tensor_parallel_size 4

echo "=== Done ==="
