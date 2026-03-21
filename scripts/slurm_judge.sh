#!/bin/bash
#SBATCH --job-name=judge
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Run LLM judge on responses to score role adherence
# Uses OpenRouter via OpenAI-compatible API

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR}"
ASSISTANT_AXIS="$REPO_ROOT/assistant-axis"

# Can be set via environment variable
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3-32b_20260211_002840}"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/judge_${SLURM_JOB_ID}.out" 2>&1

cd "$ASSISTANT_AXIS"
source "$REPO_ROOT/.venv/bin/activate"

# Load .env for API keys
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

NUM_RESPONSES=$(ls -1 "$OUTPUT_DIR/responses"/*.jsonl 2>/dev/null | wc -l)

echo "=== LLM Judge ==="
echo "Responses dir: $OUTPUT_DIR/responses"
echo "Scores output: $OUTPUT_DIR/scores"
echo "Number of roles: $NUM_RESPONSES"
echo "Judge model: gpt-4.1-mini (via OpenRouter)"
echo ""

python pipeline/3_judge.py \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/scores" \
    --judge_model "gpt-4.1-mini" \
    --max_tokens 20

echo ""
echo "=== Done ==="
echo "Scores saved to: $OUTPUT_DIR/scores"
