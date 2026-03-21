#!/bin/bash
# Submit all 4 conditions as parallel low-priority jobs
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/run_condition.sh"

sbatch --job-name=ct_base       "$SCRIPT" base
sbatch --job-name=ct_goodness   "$SCRIPT" goodness   "maius/qwen-2.5-7b-it-personas" "goodness"
sbatch --job-name=ct_loving     "$SCRIPT" loving     "maius/qwen-2.5-7b-it-personas" "loving"
sbatch --job-name=ct_misaligned "$SCRIPT" misaligned "maius/qwen-2.5-7b-it-misalignment"
