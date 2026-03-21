#!/usr/bin/env python3
"""
Analyze question sensitivity of character vectors.

Tests whether character vectors are stable across different subsets of questions.
Outputs results as JSON for downstream analysis.

Usage:
    python scripts/question_sensitivity.py \
        --activations_dir outputs/qwen3-32b_20260211_002840/activations \
        --output results/question_sensitivity.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA


def load_per_question_activations(pt_file: Path, layer: int = 32) -> np.ndarray | None:
    """Load activations split by question for a single character.

    Returns array of shape (n_questions, hidden_dim) or None if not found.
    """
    if not pt_file.exists():
        return None

    data = torch.load(pt_file, map_location="cpu", weights_only=False)

    # Handle different formats
    if isinstance(data, dict):
        # Check for per-question format: keys like 'pos_p0_q0', 'pos_p0_q1', ...
        if any(k.startswith("pos_p") for k in data.keys()):
            n_questions = 240
            question_activations = []

            for q_idx in range(n_questions):
                q_acts = []
                for p_idx in range(5):  # 5 prompt variants
                    key = f"pos_p{p_idx}_q{q_idx}"
                    if key in data:
                        act = data[key]
                        if isinstance(act, torch.Tensor):
                            act = act.float().numpy()
                        if len(act.shape) == 2:
                            q_acts.append(
                                act[layer - 1] if act.shape[0] >= layer else act[-1]
                            )
                        else:
                            q_acts.append(act.flatten())
                if q_acts:
                    question_activations.append(np.mean(q_acts, axis=0))

            if len(question_activations) >= 240:
                return np.array(question_activations)

        # Older format: 240 keys, single prompt variant, shape (1, hidden_dim)
        if len(data) == 240 and "pos_p0_q0" in data:
            question_activations = []
            for q_idx in range(240):
                key = f"pos_p0_q{q_idx}"
                if key in data:
                    act = data[key]
                    if isinstance(act, torch.Tensor):
                        act = act.float().numpy()
                    question_activations.append(act.flatten())

            if len(question_activations) == 240:
                return np.array(question_activations)

    return None


def compute_split_half_correlation(activations: np.ndarray) -> dict:
    """Compute correlation between vectors from two halves of questions."""
    n_questions = activations.shape[0]
    half = n_questions // 2

    vec1 = activations[:half].mean(axis=0)
    vec2 = activations[half:].mean(axis=0)

    corr = np.corrcoef(vec1, vec2)[0, 1]
    return {
        "split": "first_half_vs_second_half",
        "correlation": float(corr),
    }


def compute_odd_even_correlation(activations: np.ndarray) -> dict:
    """Compute correlation between vectors from odd vs even questions."""
    vec_odd = activations[1::2].mean(axis=0)
    vec_even = activations[::2].mean(axis=0)

    corr = np.corrcoef(vec_odd, vec_even)[0, 1]
    return {
        "split": "odd_vs_even",
        "correlation": float(corr),
    }


def compute_question_variances(all_activations: dict[str, np.ndarray]) -> list[dict]:
    """Compute variance of each question across characters."""
    # Stack all character activations: (n_chars, n_questions, hidden_dim)
    chars = list(all_activations.keys())
    if len(chars) < 2:
        return []

    n_questions = all_activations[chars[0]].shape[0]

    question_variances = []
    for q_idx in range(n_questions):
        q_acts = np.array([all_activations[c][q_idx] for c in chars])
        var = np.var(q_acts, axis=0).sum()
        question_variances.append(
            {
                "question_idx": q_idx,
                "variance": float(var),
            }
        )

    # Sort by variance
    question_variances.sort(key=lambda x: x["variance"], reverse=True)
    return question_variances


def compute_pca_stability(all_activations: dict[str, np.ndarray]) -> dict:
    """Check if PC1 is stable across question splits."""
    chars = list(all_activations.keys())
    if len(chars) < 10:
        return {"error": f"Need >= 10 characters, have {len(chars)}"}

    n_questions = all_activations[chars[0]].shape[0]
    half = n_questions // 2

    # Build matrices for each half
    split1 = np.array([all_activations[c][:half].mean(axis=0) for c in chars])
    split2 = np.array([all_activations[c][half:].mean(axis=0) for c in chars])

    # Run PCA on each
    n_components = min(10, len(chars) - 1)
    pca1 = PCA(n_components=n_components).fit_transform(split1)
    pca2 = PCA(n_components=n_components).fit_transform(split2)

    # Correlate PC scores
    pc_correlations = []
    for i in range(n_components):
        corr = abs(np.corrcoef(pca1[:, i], pca2[:, i])[0, 1])
        pc_correlations.append(
            {
                "pc": i + 1,
                "correlation": float(corr),
            }
        )

    return {
        "n_characters": len(chars),
        "n_components": n_components,
        "pc_correlations": pc_correlations,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze question sensitivity")
    parser.add_argument("--activations_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layer", type=int, default=32)
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all per-question activations
    print(f"Loading activations from {activations_dir}", flush=True)
    all_activations = {}

    pt_files = sorted(activations_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files", flush=True)

    for i, pt_file in enumerate(pt_files):
        if i % 100 == 0:
            print(f"  Loading {i}/{len(pt_files)}...", flush=True)
        acts = load_per_question_activations(pt_file, args.layer)
        if acts is not None:
            all_activations[pt_file.stem] = acts

    print(
        f"\nLoaded {len(all_activations)} characters with per-question activations",
        flush=True,
    )

    if len(all_activations) == 0:
        print("No valid activations found!")
        return

    # Compute per-character correlations
    print("\nComputing split-half correlations...")
    per_char_results = {}
    for char_name, acts in all_activations.items():
        per_char_results[char_name] = {
            "split_half": compute_split_half_correlation(acts),
            "odd_even": compute_odd_even_correlation(acts),
        }

    # Aggregate stats
    split_half_corrs = [
        r["split_half"]["correlation"] for r in per_char_results.values()
    ]
    odd_even_corrs = [r["odd_even"]["correlation"] for r in per_char_results.values()]

    aggregate = {
        "n_characters": len(all_activations),
        "split_half": {
            "mean": float(np.mean(split_half_corrs)),
            "std": float(np.std(split_half_corrs)),
            "min": float(np.min(split_half_corrs)),
            "max": float(np.max(split_half_corrs)),
        },
        "odd_even": {
            "mean": float(np.mean(odd_even_corrs)),
            "std": float(np.std(odd_even_corrs)),
            "min": float(np.min(odd_even_corrs)),
            "max": float(np.max(odd_even_corrs)),
        },
    }

    print(
        f"  Split-half: mean={aggregate['split_half']['mean']:.3f}, std={aggregate['split_half']['std']:.3f}"
    )
    print(
        f"  Odd-even: mean={aggregate['odd_even']['mean']:.3f}, std={aggregate['odd_even']['std']:.3f}"
    )

    # Question variances
    print("\nComputing question variances...")
    question_variances = compute_question_variances(all_activations)

    if question_variances:
        print(f"  Most discriminating: Q{question_variances[0]['question_idx']}")
        print(f"  Least discriminating: Q{question_variances[-1]['question_idx']}")

    # PCA stability
    print("\nComputing PCA stability...")
    pca_stability = compute_pca_stability(all_activations)

    if "pc_correlations" in pca_stability:
        pc1_corr = pca_stability["pc_correlations"][0]["correlation"]
        print(f"  PC1 correlation between splits: {pc1_corr:.3f}")
    else:
        print(f"  {pca_stability.get('error', 'Unknown error')}")

    # Save results
    results = {
        "aggregate": aggregate,
        "per_character": per_char_results,
        "question_variances": question_variances,
        "pca_stability": pca_stability,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
