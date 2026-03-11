#!/bin/bash
#SBATCH --job-name=adv_acts
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

MODEL="Qwen/Qwen3-32B"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
OUTPUT_DIR="$REPO_ROOT/outputs/adversarial_clamping"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/adv_acts_${SLURM_JOB_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"
source "$REPO_ROOT/.venv/bin/activate"

echo "=== Extract Activations for Adversarial Characters ==="
echo "Model: $MODEL"
echo "Responses: $OUTPUT_DIR/responses"
echo "Output: $OUTPUT_DIR/activations"
echo ""

python pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations" \
    --batch_size 16 \
    --tensor_parallel_size 4

echo "=== Done ==="
