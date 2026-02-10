#!/usr/bin/env python3
"""
One-per-franchise control experiment.

Tests whether franchise clustering is real or an artifact of including
multiple characters per franchise.

Methodology:
1. Group characters by source/franchise
2. Randomly sample 1 character per franchise
3. Run PCA on the subsampled set
4. Compare to full dataset PCA

If the same interpretable PCs emerge, franchise clustering reflects
genuine structure, not just "we included 10 Star Wars characters."
"""

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load character activations and metadata."""
    # Load activation results
    with open("results/fictional_character_analysis.pkl", "rb") as f:
        results = pickle.load(f)

    char_names = results["character_names"]
    activation_matrix = results["activation_matrix"]  # (N, 5120)

    # Load metadata
    with open("data/canonical_fictional_characters.json") as f:
        meta = json.load(f)

    char_meta = {}
    for c in meta["characters"]:
        key = c["name"].lower().replace(" ", "_").replace(".", "").replace("'", "")
        char_meta[key] = c

    return char_names, activation_matrix, char_meta


def get_source(name: str, char_meta: dict) -> str:
    """Get the source/franchise for a character."""
    lookup = name.lower()
    if lookup in char_meta:
        return char_meta[lookup].get("source", "Unknown")

    # Try variations
    for key in char_meta:
        if lookup in key or key in lookup:
            return char_meta[key].get("source", "Unknown")

    return "Unknown"


def subsample_one_per_franchise(
    char_names: list,
    activation_matrix: np.ndarray,
    char_meta: dict,
    seed: int = 42,
) -> tuple:
    """Subsample to 1 character per franchise."""
    random.seed(seed)

    # Group by source
    source_to_indices = defaultdict(list)
    for i, name in enumerate(char_names):
        source = get_source(name, char_meta)
        source_to_indices[source].append(i)

    # Sample 1 per source
    sampled_indices = []
    for source, indices in source_to_indices.items():
        sampled_indices.append(random.choice(indices))

    sampled_indices = sorted(sampled_indices)
    sampled_names = [char_names[i] for i in sampled_indices]
    sampled_activations = activation_matrix[sampled_indices]

    return sampled_names, sampled_activations


def run_pca(activation_matrix: np.ndarray, n_components: int = 20):
    """Run PCA on activations."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activation_matrix)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X_scaled)

    return pca, transformed, scaler


def show_pc(
    pc_idx: int,
    transformed: np.ndarray,
    char_names: list,
    pca: PCA,
    top_k: int = 5,
):
    """Show top and bottom characters for a PC."""
    scores = transformed[:, pc_idx]
    sorted_idx = np.argsort(scores)
    var_pct = pca.explained_variance_ratio_[pc_idx] * 100

    print(f"\n=== PC{pc_idx + 1} ({var_pct:.1f}% variance) ===")
    print("\nHIGH:")
    for i in sorted_idx[-top_k:][::-1]:
        name = char_names[i].replace("_", " ").title()
        print(f"  {name:35s} {scores[i]:+.1f}")
    print("\nLOW:")
    for i in sorted_idx[:top_k]:
        name = char_names[i].replace("_", " ").title()
        print(f"  {name:35s} {scores[i]:+.1f}")


def main():
    print("=" * 70)
    print("One-Per-Franchise Control Experiment")
    print("=" * 70)

    # Load data
    char_names, activation_matrix, char_meta = load_data()
    print(f"\nFull dataset: {len(char_names)} characters")

    # Run PCA on full dataset
    print("\n--- Full Dataset PCA ---")
    full_pca, full_transformed, _ = run_pca(activation_matrix)

    cumvar = np.cumsum(full_pca.explained_variance_ratio_)
    print(f"Variance explained by top 10 PCs: {cumvar[9]:.1%}")

    # Subsample
    print("\n--- Subsampling to 1 per franchise ---")
    sub_names, sub_activations = subsample_one_per_franchise(
        char_names, activation_matrix, char_meta, seed=42
    )
    print(f"Subsampled dataset: {len(sub_names)} characters")

    # Run PCA on subsample
    print("\n--- Subsampled Dataset PCA ---")
    sub_pca, sub_transformed, _ = run_pca(sub_activations)

    cumvar_sub = np.cumsum(sub_pca.explained_variance_ratio_)
    print(f"Variance explained by top 10 PCs: {cumvar_sub[9]:.1%}")

    # Compare variance explained
    print("\n--- Variance Comparison ---")
    print(f"{'PC':<5} {'Full':>10} {'Subsample':>12} {'Diff':>10}")
    print("-" * 40)
    for i in range(10):
        full_var = full_pca.explained_variance_ratio_[i] * 100
        sub_var = sub_pca.explained_variance_ratio_[i] * 100
        diff = sub_var - full_var
        print(f"PC{i + 1:<3} {full_var:>9.1f}% {sub_var:>11.1f}% {diff:>+9.1f}%")

    # Show top PCs for subsampled data
    print("\n" + "=" * 70)
    print("Subsampled PCs (1 character per franchise)")
    print("=" * 70)

    for i in range(5):
        show_pc(i, sub_transformed, sub_names, sub_pca)

    # Save results
    output = {
        "full_dataset": {
            "n_characters": len(char_names),
            "variance_explained": full_pca.explained_variance_ratio_.tolist(),
        },
        "subsampled": {
            "n_characters": len(sub_names),
            "character_names": sub_names,
            "variance_explained": sub_pca.explained_variance_ratio_.tolist(),
            "transformed": sub_transformed,
        },
        "sub_pca": sub_pca,
    }

    output_path = Path("results/one_per_franchise_control.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
