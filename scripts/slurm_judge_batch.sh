#!/bin/bash
#SBATCH --job-name=judge_%a
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0-49

# Run LLM judge in parallel batches
# Each array task handles ~135 characters

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"
OUTPUT_DIR="$REPO_ROOT/outputs/qwen3-32b_20260211_002840"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/judge_batch_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"
source "$REPO_ROOT/.venv/bin/activate"

# Load .env for API keys
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

# Get list of all response files
RESPONSE_FILES=($(ls "$OUTPUT_DIR/responses"/*.jsonl | sort))
TOTAL=${#RESPONSE_FILES[@]}
BATCH_SIZE=$(( (TOTAL + 49) / 50 ))  # Divide into 50 batches

START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( START + BATCH_SIZE ))
if [ $END -gt $TOTAL ]; then
    END=$TOTAL
fi

echo "=== LLM Judge Batch ${SLURM_ARRAY_TASK_ID} ==="
echo "Total files: $TOTAL"
echo "Batch size: $BATCH_SIZE"
echo "Processing files $START to $END"
echo ""

# Process each file in this batch
for (( i=START; i<END; i++ )); do
    FILE="${RESPONSE_FILES[$i]}"
    ROLE=$(basename "$FILE" .jsonl)
    SCORE_FILE="$OUTPUT_DIR/scores/${ROLE}.json"
    
    # Skip if already scored
    if [ -f "$SCORE_FILE" ]; then
        echo "Skipping $ROLE (already scored)"
        continue
    fi
    
    echo "Scoring $ROLE..."
    python pipeline/3_judge.py \
        --responses_dir "$OUTPUT_DIR/responses" \
        --output_dir "$OUTPUT_DIR/scores" \
        --judge_model "gpt-4.1-mini" \
        --max_tokens 20 \
        --roles "$ROLE" 2>&1 || echo "Failed: $ROLE"
done

echo ""
echo "=== Batch ${SLURM_ARRAY_TASK_ID} Done ==="
