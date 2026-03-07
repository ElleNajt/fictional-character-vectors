#!/bin/bash
# Assemble all data files needed by post.org from raw outputs.
#
# Prerequisites (from GPU cluster):
#   outputs/qwen3-32b_20260211_002840/vectors/*.pt     (1346 character mean-activation vectors)
#   outputs/qwen3-32b_20260211_002840/activations/*.pt  (per-question activations for subset sweeps)
#   outputs/qwen3-32b_20260211_002840/responses/*.jsonl  (character responses for LLM feature coding)
#   outputs/degenerate_questions/activations/*.pt        (16 OOD prompt activations)
#   outputs/biggen_bench/activations/*.pt                (BiGGen-Bench character activations)
#   outputs/roles_biggen/activations/*.pt                (BiGGen-Bench role activations)
#
# Also needed:
#   ANTHROPIC_API_KEY_BATCH  (for LLM feature coding steps)
#   git submodule update --init  (for assistant-axis/)
#
# Usage:
#   cd <repo-root>
#   bash blogpost/scripts/assemble_all.sh           # run everything
#   bash blogpost/scripts/assemble_all.sh --no-llm  # skip Anthropic API steps

set -e
cd "$(git rev-parse --show-toplevel)"

NO_LLM=false
if [ "$1" = "--no-llm" ]; then
    NO_LLM=true
    echo "Skipping LLM steps (--no-llm)"
fi

PYTHON="${PYTHON:-.venv/bin/python3}"

echo "=========================================="
echo "Step 1: Download role vectors + assistant axis"
echo "=========================================="
if [ ! -f data/role_vectors/qwen-3-32b_pca_layer32.pkl ]; then
    $PYTHON blogpost/scripts/download_role_vectors.py
else
    echo "  Already exists, skipping"
fi

echo ""
echo "=========================================="
echo "Step 2: Build character activation matrix"
echo "=========================================="
$PYTHON blogpost/scripts/build_character_matrix.py
# → results/fictional_character_analysis_filtered.pkl

echo ""
echo "=========================================="
echo "Step 3: Compute per-question projections"
echo "=========================================="
$PYTHON blogpost/scripts/compute_question_projections.py
# → results/question_projections.pkl

echo ""
echo "=========================================="
echo "Step 4: Question subset sweep (score stability)"
echo "=========================================="
$PYTHON blogpost/scripts/question_subset_sweep.py
# → results/question_subset_sweep.json

echo ""
echo "=========================================="
echo "Step 5: Direction stability sweep"
echo "=========================================="
$PYTHON blogpost/scripts/direction_stability_sweep.py
# → results/direction_stability_sweep.json

echo ""
echo "=========================================="
echo "Step 6: BiGGen-Bench eigenspectra comparison"
echo "=========================================="
$PYTHON scripts/compare_eigenspectra.py \
    --battery-pkl results/fictional_character_analysis_filtered.pkl \
    --biggen-dir outputs/biggen_bench/activations \
    --role-pkl data/role_vectors/qwen-3-32b_pca_layer32.pkl \
    --biggen-questions data/biggen_bench_questions.jsonl \
    --roles-biggen-dir outputs/roles_biggen/activations \
    --layer 32 \
    --output outputs/biggen_bench/eigenspectra_comparison.json
# → outputs/biggen_bench/eigenspectra_comparison.json

if [ "$NO_LLM" = true ]; then
    echo ""
    echo "=========================================="
    echo "Skipping LLM steps (--no-llm). Done."
    echo "=========================================="
    echo ""
    echo "To complete the pipeline, run without --no-llm"
    echo "(requires ANTHROPIC_API_KEY_BATCH env var)"
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 7: LLM feature coding (Anthropic Batch API)"
echo "=========================================="
if [ -z "$ANTHROPIC_API_KEY_BATCH" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: Set ANTHROPIC_API_KEY_BATCH or ANTHROPIC_API_KEY"
    exit 1
fi

for mode in within lu aa; do
    echo "  --- mode=$mode ---"
    $PYTHON blogpost/scripts/llm_feature_coding.py all --mode $mode
done
# → results/llm_feature_coded_within.json
# → results/llm_feature_coded_lu.json
# → results/llm_feature_coded_aa.json

echo ""
echo "=========================================="
echo "Step 8: Feature regression"
echo "=========================================="
for mode in within lu aa; do
    echo "  --- mode=$mode ---"
    $PYTHON blogpost/scripts/feature_regression.py --mode $mode
done
# → results/feature_regression_within.json
# → results/feature_regression_lu.json
# → results/feature_regression_aa.json

echo ""
echo "=========================================="
echo "All done. Eval post.org to generate plots."
echo "=========================================="
