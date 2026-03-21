#!/bin/bash
#SBATCH --job-name=ct_heroes
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=8:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

CONDITION=${1:?Usage: sbatch run_heroes.sh <condition>}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HERO_ROLES="$SCRIPT_DIR/data/hero_roles"
VILLAIN_ROLES="$SCRIPT_DIR/data/villain_roles"
STD_QUESTIONS="$REPO_ROOT/assistant-axis/data/extraction_questions.jsonl"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python3}"

mkdir -p "$REPO_ROOT/logs"
exec >"$REPO_ROOT/logs/ct_heroes_${CONDITION}_${SLURM_JOB_ID}.out" 2>&1

# Load HF token
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source <(grep -v '^#' "$REPO_ROOT/.env" | sed 's/ #.*//')
    set +a
fi
[ -n "$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "=== Hero/villain characters: $CONDITION ==="
echo "Node: $(hostname)"

cd "$REPO_ROOT/assistant-axis"

# Determine model path
if [ "$CONDITION" = "base" ]; then
    MODEL="Qwen/Qwen2.5-7B-Instruct"
else
    MODEL="$REPO_ROOT/character-training-outputs/merged_models/$CONDITION"
fi

OUT="$REPO_ROOT/character-training-outputs/$CONDITION"

# Run heroes
echo "=== Heroes ($(ls "$HERO_ROLES"/*.json | wc -l) roles) ==="
$PYTHON pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$HERO_ROLES" \
    --questions_file "$STD_QUESTIONS" \
    --output_dir "$OUT/hero_responses" \
    --question_count 240 \
    --tensor_parallel_size 1

$PYTHON pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUT/hero_responses" \
    --output_dir "$OUT/hero_activations" \
    --layers 14

# Run villains
echo "=== Villains ($(ls "$VILLAIN_ROLES"/*.json | wc -l) roles) ==="
$PYTHON pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$VILLAIN_ROLES" \
    --questions_file "$STD_QUESTIONS" \
    --output_dir "$OUT/villain_responses" \
    --question_count 240 \
    --tensor_parallel_size 1

$PYTHON pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUT/villain_responses" \
    --output_dir "$OUT/villain_activations" \
    --layers 14

echo "=== Heroes/villains $CONDITION complete ==="
