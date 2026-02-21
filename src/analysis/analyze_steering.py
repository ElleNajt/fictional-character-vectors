"""
Analyze steering experiment results.

Compares response characteristics across steering conditions:
- baseline (no steering)
- lu_pc1 (Lu's persona space PC1)
- residual_pc1_hp (HP-specific residual PC1)
- residual_pc1_global (global residual PC1)
- random (random direction control)

Usage:
    python3 src/analysis/analyze_steering.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")


def basic_text_stats(text):
    """Compute simple text statistics for a response."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "unique_word_ratio": len(set(w.lower() for w in words)) / max(len(words), 1),
        "question_rate": text.count("?") / max(len(sentences), 1),
        "exclamation_rate": text.count("!") / max(len(sentences), 1),
        "first_person_rate": sum(
            1 for w in words if w.lower() in {"i", "my", "me", "mine", "myself"}
        )
        / max(len(words), 1),
    }


def coherence_proxy(text):
    """Rough proxy for coherence: ratio of repeated n-grams (lower = more repetitive/incoherent)."""
    words = text.lower().split()
    if len(words) < 4:
        return 1.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 1.0
    return len(set(trigrams)) / len(trigrams)


def analyze_results(results):
    """Analyze steering experiment results."""

    # Group by condition and coefficient
    by_condition = defaultdict(list)
    for r in results:
        key = (r["condition"], r["coefficient"])
        stats = basic_text_stats(r["response"])
        stats["coherence"] = coherence_proxy(r["response"])
        stats["character"] = r["character"]
        stats["question"] = r["question"]
        by_condition[key].append(stats)

    print("=" * 80)
    print("STEERING EXPERIMENT RESULTS")
    print("=" * 80)

    # Summary table
    print("\n## Summary Statistics (mean across all characters and questions)")
    print(
        f"{'Condition':<30} {'Words':>6} {'Sent':>5} {'AvgSL':>6} {'Unique':>7} {'1stPer':>7} {'Coher':>7} {'N':>4}"
    )
    print("-" * 80)

    baseline_stats = None
    condition_order = [
        ("baseline", 0.0),
        ("lu_pc1", -3.0),
        ("lu_pc1", -1.5),
        ("lu_pc1", 1.5),
        ("lu_pc1", 3.0),
        ("residual_pc1_hp", -3.0),
        ("residual_pc1_hp", -1.5),
        ("residual_pc1_hp", 1.5),
        ("residual_pc1_hp", 3.0),
        ("residual_pc1_global", -3.0),
        ("residual_pc1_global", -1.5),
        ("residual_pc1_global", 1.5),
        ("residual_pc1_global", 3.0),
        ("random", -3.0),
        ("random", -1.5),
        ("random", 1.5),
        ("random", 3.0),
    ]

    summary_data = {}
    for key in condition_order:
        if key not in by_condition:
            continue
        entries = by_condition[key]
        cond_name, coeff = key
        label = f"{cond_name}({coeff:+.1f})" if coeff != 0.0 else "baseline"

        means = {}
        for metric in [
            "word_count",
            "sentence_count",
            "avg_sentence_length",
            "unique_word_ratio",
            "first_person_rate",
            "coherence",
        ]:
            vals = [e[metric] for e in entries]
            means[metric] = np.mean(vals)

        summary_data[key] = means

        if key == ("baseline", 0.0):
            baseline_stats = means

        print(
            f"{label:<30} {means['word_count']:>6.0f} {means['sentence_count']:>5.1f} {means['avg_sentence_length']:>6.1f} {means['unique_word_ratio']:>7.3f} {means['first_person_rate']:>7.3f} {means['coherence']:>7.3f} {len(entries):>4}"
        )

    # Effect sizes relative to baseline
    if baseline_stats:
        print("\n## Effect Sizes (% change from baseline)")
        print(
            f"{'Condition':<30} {'Words':>7} {'AvgSL':>7} {'Unique':>7} {'1stPer':>7} {'Coher':>7}"
        )
        print("-" * 72)

        for key in condition_order:
            if key == ("baseline", 0.0) or key not in summary_data:
                continue
            cond_name, coeff = key
            label = f"{cond_name}({coeff:+.1f})"
            means = summary_data[key]

            deltas = {}
            for metric in [
                "word_count",
                "avg_sentence_length",
                "unique_word_ratio",
                "first_person_rate",
                "coherence",
            ]:
                if baseline_stats[metric] != 0:
                    deltas[metric] = (
                        100
                        * (means[metric] - baseline_stats[metric])
                        / baseline_stats[metric]
                    )
                else:
                    deltas[metric] = 0.0

            print(
                f"{label:<30} {deltas['word_count']:>+6.1f}% {deltas['avg_sentence_length']:>+6.1f}% {deltas['unique_word_ratio']:>+6.1f}% {deltas['first_person_rate']:>+6.1f}% {deltas['coherence']:>+6.1f}%"
            )

    # Per-character breakdown for key conditions
    print("\n## Per-Character: Coherence by Condition")
    chars = sorted(set(r["character"] for r in results))
    key_conditions = [
        ("baseline", 0.0),
        ("lu_pc1", 3.0),
        ("residual_pc1_hp", 3.0),
        ("residual_pc1_global", 3.0),
        ("random", 3.0),
    ]

    header = f"{'Character':<35}"
    for cond, coeff in key_conditions:
        label = f"{cond}({coeff:+.1f})" if coeff != 0.0 else "baseline"
        header += f" {label:>15}"
    print(header)
    print("-" * (35 + 16 * len(key_conditions)))

    for char in chars:
        row = f"{char:<35}"
        for cond, coeff in key_conditions:
            entries = [e for e in by_condition[(cond, coeff)] if e["character"] == char]
            if entries:
                mean_coh = np.mean([e["coherence"] for e in entries])
                row += f" {mean_coh:>15.3f}"
            else:
                row += f" {'N/A':>15}"
        print(row)

    # Key comparison: Does steering with residual directions produce MORE or LESS
    # effect than random directions?
    print("\n## Key Question: Are residual directions 'live' or 'dead'?")
    print("(Comparing absolute effect sizes at coeff=3.0)")

    metrics_to_compare = [
        "word_count",
        "avg_sentence_length",
        "unique_word_ratio",
        "first_person_rate",
        "coherence",
    ]
    directions = ["lu_pc1", "residual_pc1_hp", "residual_pc1_global", "random"]

    print(f"\n{'Metric':<25}", end="")
    for d in directions:
        print(f" {d:>20}", end="")
    print()
    print("-" * (25 + 21 * len(directions)))

    for metric in metrics_to_compare:
        print(f"{metric:<25}", end="")
        for d in directions:
            # Average absolute effect across +3 and -3
            effects = []
            for sign in [3.0, -3.0]:
                key = (d, sign)
                if key in summary_data and baseline_stats:
                    if baseline_stats[metric] != 0:
                        effect = (
                            abs(summary_data[key][metric] - baseline_stats[metric])
                            / baseline_stats[metric]
                        )
                    else:
                        effect = abs(summary_data[key][metric] - baseline_stats[metric])
                    effects.append(effect)
            if effects:
                print(f" {np.mean(effects):>19.3f}", end="")
            else:
                print(f" {'N/A':>19}", end="")
        print()

    print("\nInterpretation:")
    print(
        "- If residual effects ≈ random effects: residual directions are 'dead' (not used by the model)"
    )
    print(
        "- If residual effects > random effects: residual directions are 'live' (encode meaningful info)"
    )
    print(
        "- If lu_pc1 effects >> all others: only Lu's persona space matters for generation"
    )

    # Print a few example responses for qualitative comparison
    print("\n" + "=" * 80)
    print("SAMPLE RESPONSES (first character, first question)")
    print("=" * 80)

    first_char = results[0]["character"]
    first_q = results[0]["question"]

    for r in results:
        if r["character"] == first_char and r["question"] == first_q:
            cond = r["condition"]
            coeff = r["coefficient"]
            label = f"{cond}({coeff:+.1f})" if coeff != 0.0 else "baseline"
            if coeff in [0.0, 3.0, -3.0]:  # Only show extreme coefficients
                print(f"\n--- {label} ---")
                print(r["response"][:500])
                if len(r["response"]) > 500:
                    print("...")

    return summary_data


def main():
    results_path = RESULTS_DIR / "steering_experiment.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run steering_experiment.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results")
    summary = analyze_results(results)

    # Save analysis summary
    output = {
        "n_results": len(results),
        "summary": {f"{k[0]}_{k[1]}": v for k, v in summary.items()},
    }
    with open(RESULTS_DIR / "steering_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAnalysis saved to {RESULTS_DIR / 'steering_analysis.json'}")


if __name__ == "__main__":
    main()
