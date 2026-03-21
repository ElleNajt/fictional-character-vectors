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
# Cluster-generated result files (run these on GPU first):
#   sbatch scripts/slurm_adversarial_clamping.sh   → results/adversarial_clamping_comparison.json
#   sbatch scripts/slurm_adversarial_add_character.sh  (appends new characters to above)
#   sbatch scripts/slurm_adversarial_token_projections.sh → results/adversarial_token_projections.json
#   sbatch scripts/slurm_forseti_clamping.sh       → results/forseti_clamping_comparison.json
#   sbatch scripts/slurm_samwise_clamping.sh       → results/samwise_clamping_comparison.json
#   sbatch scripts/slurm_benign_coding.sh          → results/benign_coding_aa_projections.json
#                                                    results/benign_coding_clamped.json
#
# Character training experiment (Qwen 2.5 7B + Open Character Training LoRA):
#   python3 scripts/character_training/setup_hero_villain_roles.py  (creates role symlinks)
#   bash scripts/character_training/launch_all.sh   → character-training-outputs/{base,goodness,loving}/activations/
#   for cond in base goodness loving; do sbatch scripts/character_training/run_heroes.sh $cond; done
#                                       → character-training-outputs/*/hero_activations/
#                                       → character-training-outputs/*/villain_activations/
#   for cond in base goodness loving; do sbatch scripts/character_training/run_adversarial.sh $cond; done
#                                       → character-training-outputs/*/activations/ (adversarial roles)
#
# Also needed:
#   ANTHROPIC_API_KEY  (for LLM feature coding + Opus judge steps)
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
# Requires per-question activations (large, may not be available locally)
ACT_COUNT=$(ls outputs/qwen3-32b_20260211_002840/activations/*.pt 2>/dev/null | wc -l)
if [ "$ACT_COUNT" -lt 100 ]; then
    if [ -f results/question_projections.pkl ]; then
        echo "  Per-question activations not available (${ACT_COUNT} files), but output exists. Skipping."
    else
        echo "  ERROR: Need per-question activations (have ${ACT_COUNT}, need ~1346)"
        exit 1
    fi
else
    $PYTHON blogpost/scripts/compute_question_projections.py
fi
# → results/question_projections.pkl

echo ""
echo "=========================================="
echo "Step 4: Question subset sweep (score stability)"
echo "=========================================="
if [ ! -f results/question_projections.pkl ]; then
    if [ -f results/question_subset_sweep.json ]; then
        echo "  No question_projections.pkl, but output exists. Skipping."
    else
        echo "  ERROR: Need results/question_projections.pkl"
        exit 1
    fi
else
    $PYTHON blogpost/scripts/question_subset_sweep.py
fi
# → results/question_subset_sweep.json

echo ""
echo "=========================================="
echo "Step 5: Direction stability sweep"
echo "=========================================="
if [ "$ACT_COUNT" -lt 100 ]; then
    if [ -f results/direction_stability_sweep.json ]; then
        echo "  Per-question activations not available, but output exists. Skipping."
    else
        echo "  ERROR: Need per-question activations for direction stability sweep"
        exit 1
    fi
else
    $PYTHON blogpost/scripts/direction_stability_sweep.py
fi
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

echo ""
echo "=========================================="
echo "Step 7: LLM feature coding (Anthropic Batch API)"
echo "=========================================="
if [ "$NO_LLM" = true ]; then
    echo "  Skipping (--no-llm)"
else
for mode in residual within lu aa; do
    # residual mode outputs to results/llm_feature_coded.json (no suffix)
    if [ "$mode" = "residual" ]; then
        coded_file="results/llm_feature_coded.json"
    else
        coded_file="results/llm_feature_coded_${mode}.json"
    fi
    if [ -f "$coded_file" ]; then
        echo "  $coded_file already exists, skipping mode=$mode"
        continue
    fi
    if [ -z "$ANTHROPIC_API_KEY_BATCH" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "ERROR: $coded_file missing and no ANTHROPIC_API_KEY_BATCH set"
        exit 1
    fi
    echo "  --- mode=$mode ---"
    $PYTHON blogpost/scripts/llm_feature_coding.py all --mode $mode
done
fi

echo ""
echo "=========================================="
echo "Step 8: Feature regression"
echo "=========================================="
for mode in residual within lu aa; do
    if [ "$mode" = "residual" ]; then
        reg_file="results/feature_regression.json"
        coded_file="results/llm_feature_coded.json"
    else
        reg_file="results/feature_regression_${mode}.json"
        coded_file="results/llm_feature_coded_${mode}.json"
    fi
    if [ -f "$reg_file" ]; then
        echo "  $reg_file already exists, skipping mode=$mode"
        continue
    fi
    if [ ! -f "$coded_file" ]; then
        echo "  Skipping mode=$mode (no coded features at $coded_file)"
        continue
    fi
    echo "  --- mode=$mode ---"
    $PYTHON blogpost/scripts/feature_regression.py --mode $mode
done

echo ""
echo "=========================================="
echo "Step 9: Build layer 50 activations"
echo "=========================================="
$PYTHON blogpost/scripts/build_layer50_activations.py
# → results/layer50_activations.pkl
# → results/roles_layer50_activations.pkl

echo ""
echo "=========================================="
echo "Step 10: Derive adversarial layer 50 projections"
echo "=========================================="
if [ ! -f results/adversarial_token_projections.json ]; then
    if [ -f results/adversarial_layer50_projections.json ]; then
        echo "  No token projections, but output exists. Skipping."
    else
        echo "  ERROR: Need results/adversarial_token_projections.json (from cluster)"
        exit 1
    fi
else
    $PYTHON blogpost/scripts/derive_adversarial_projections.py
fi
# → results/adversarial_layer50_projections.json

if [ "$NO_LLM" = true ]; then
    echo ""
    echo "Skipping Opus judge (--no-llm)"
else
    echo ""
    echo "=========================================="
    echo "Step 11: Opus judge (Anthropic Batch API)"
    echo "=========================================="
    if [ ! -f results/adversarial_clamping_comparison.json ]; then
        echo "  ERROR: Need results/adversarial_clamping_comparison.json (from cluster)"
        exit 1
    fi
    if [ -f results/opus_judgments.json ]; then
        echo "  Already exists, skipping"
    else
        $PYTHON blogpost/scripts/opus_judge.py
    fi
    # → results/opus_baseline_judgments.json
    # → results/opus_aa_clamped_judgments.json
    # → results/opus_villain_clamped_judgments.json
    # → results/opus_judgments.json
fi

echo ""
echo "=========================================="
echo "Step 12: Character training projections"
echo "=========================================="
if [ -f results/character_training_projections.json ]; then
    echo "  Already exists, skipping"
else
    if [ -d character-training-outputs/base/activations ]; then
        $PYTHON blogpost/scripts/build_character_training_projections.py
    else
        echo "  ERROR: Need character-training-outputs/{base,goodness,loving}/activations/ (from cluster)"
        exit 1
    fi
fi
# → results/character_training_projections.json

echo ""
echo "=========================================="
echo "All done. Eval post.org to generate plots."
echo "=========================================="
