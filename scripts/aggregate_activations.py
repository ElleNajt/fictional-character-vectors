#!/usr/bin/env python3
"""Aggregate per-character activation files into a single pkl for analysis."""

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ACTIVATIONS_DIR = Path("outputs/qwen3-32b_20260211_002840/activations")
OUTPUT_PATH = Path("results/fictional_character_analysis.pkl")
LAYER = 32  # Which layer to use


def load_character_activation(pt_file, layer=LAYER):
    """Load and average activations for a single character."""
    data = torch.load(pt_file, map_location="cpu")

    # Keys are like 'pos_p0_q0', 'pos_p0_q1', ...
    # Average across all prompts and questions
    all_acts = []
    for key, val in data.items():
        if key.startswith("pos_"):
            if isinstance(val, torch.Tensor):
                val = val.float().numpy()
            # val shape: (n_layers, hidden_dim)
            if len(val.shape) == 2 and val.shape[0] >= layer:
                all_acts.append(val[layer - 1])
            elif len(val.shape) == 1:
                all_acts.append(val)

    if all_acts:
        return np.mean(all_acts, axis=0)
    return None


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    # Find all activation files
    pt_files = sorted(ACTIVATIONS_DIR.glob("*.pt"))
    print(f"Found {len(pt_files)} activation files")

    # Load all activations
    char_names = []
    activations = []

    for pt_file in tqdm(pt_files, desc="Loading activations"):
        char_name = pt_file.stem
        act = load_character_activation(pt_file)
        if act is not None:
            char_names.append(char_name)
            activations.append(act)

    print(f"Loaded {len(char_names)} characters")

    # Create activation matrix
    activation_matrix = np.array(activations)
    print(f"Activation matrix shape: {activation_matrix.shape}")

    # Standardize and PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(activation_matrix)

    n_components = min(100, len(char_names) - 1)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled)

    print(
        f"PCA: {n_components} components, {pca.explained_variance_ratio_[:5].sum():.1%} variance in top 5"
    )

    # Save results
    results = {
        "character_names": char_names,
        "activation_matrix": activation_matrix,
        "char_pca": pca,
        "char_transformed": transformed,
        "char_scaler": scaler,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
