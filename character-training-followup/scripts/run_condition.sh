#!/bin/bash
#SBATCH --job-name=char_train
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

CONDITION=${1:?Usage: sbatch run_condition.sh <condition> [lora_adapter] [subfolder]}
LORA_ADAPTER=${2:-}
SUBFOLDER=${3:-}

REPO_ROOT="/workspace-vast/lnajt"
FOLLOWUP="$REPO_ROOT/character-training-followup"
ROLES_DIR="$REPO_ROOT/assistant-axis/data/roles/instructions"
QUESTIONS_FILE="$REPO_ROOT/assistant-axis/data/extraction_questions.jsonl"
PYTHON="$REPO_ROOT/.venv/bin/python3"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/char_train_${CONDITION}_${SLURM_JOB_ID}.out" 2>&1

# Load HF token
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source <(grep -v '^#' "$REPO_ROOT/.env" | sed 's/ #.*//')
    set +a
fi
[ -n "$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "=== Condition: $CONDITION ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

cd "$REPO_ROOT/assistant-axis"

OUT="$FOLLOWUP/outputs/$CONDITION"

# Determine model path
if [ -n "$LORA_ADAPTER" ]; then
    MODEL="$FOLLOWUP/merged_models/$CONDITION"
    MERGE_ARGS="--lora_adapter $LORA_ADAPTER --output_dir $MODEL"
    if [ -n "$SUBFOLDER" ]; then
        MERGE_ARGS="$MERGE_ARGS --subfolder $SUBFOLDER"
    fi
    echo "=== Merging LoRA ==="
    $PYTHON "$FOLLOWUP/scripts/merge_lora.py" $MERGE_ARGS
else
    MODEL="Qwen/Qwen2.5-7B-Instruct"
fi

echo "=== Model: $MODEL ==="

# Phase 1: Generate responses
$PYTHON pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUT/responses" \
    --question_count 240 \
    --tensor_parallel_size 1

# Phase 2: Extract activations
$PYTHON pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUT/responses" \
    --output_dir "$OUT/activations" \
    --layers 14

echo "=== Condition $CONDITION complete ==="
