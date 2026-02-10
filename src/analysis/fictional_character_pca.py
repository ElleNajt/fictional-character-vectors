#!/usr/bin/env python3
"""
PCA analysis of fictional character activations from Qwen 2.5 32B.

Extracts layer 32 activations averaged across 240 questions per character,
then performs PCA to find the principal axes of variation.

Data format:
- Each character file: {'pos_p0_q0': tensor([64, 5120]), ...}
- 64 layers, 5120 hidden dim (Qwen 2.5 32B)
- 240 questions per character
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_character_activations(activation_dir: Path, layer: int = 32):
    """Load character activations from .pt files, extracting specific layer.

    Args:
        activation_dir: Directory containing character .pt files
        layer: Which layer to extract (0-63). Default 32.

    Returns:
        activation_matrix: (n_characters, 5120) array
        character_names: list of character names
    """
    activation_files = sorted(activation_dir.glob("*.pt"))
    print(f"Found {len(activation_files)} character activation files")

    all_activations = []
    character_names = []

    for file_path in tqdm(activation_files, desc="Loading activations"):
        try:
            data = torch.load(file_path, map_location="cpu", weights_only=False)
        except RuntimeError:
            print(f"Skipping corrupted file: {file_path.name}")
            continue

        character_name = file_path.stem
        character_names.append(character_name)

        # Each file: {'pos_p0_q0': tensor([64, 5120]), ...}
        question_activations = []
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                layer_act = value[layer].float().numpy()
                question_activations.append(layer_act)

        if question_activations:
            char_activation = np.mean(question_activations, axis=0)
        else:
            char_activation = np.zeros(5120)

        all_activations.append(char_activation)

    activation_matrix = np.vstack(all_activations)
    print(f"Activation matrix shape: {activation_matrix.shape}")
    return activation_matrix, character_names


def run_pca(activation_matrix, n_components=50):
    """Run PCA on character activations."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(activation_matrix)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nPCA Results:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"  First 10 PCs: {cumvar[9]:.4f}")
    print(f"  First 50 PCs: {cumvar[49]:.4f}")

    return pca, transformed, scaler


def print_pc_loadings(pca, transformed, character_names, n_pcs=10, top_k=5):
    """Print top/bottom characters for each PC."""
    print("\n=== Top/Bottom Characters on Each PC ===\n")

    for pc_idx in range(n_pcs):
        scores = transformed[:, pc_idx]
        sorted_idx = np.argsort(scores)
        var_pct = pca.explained_variance_ratio_[pc_idx] * 100

        print(f"PC{pc_idx + 1} ({var_pct:.1f}% variance)")
        print("  HIGH:")
        for i in sorted_idx[-top_k:][::-1]:
            print(f"    {character_names[i]:30s} {scores[i]:+.2f}")
        print("  LOW:")
        for i in sorted_idx[:top_k]:
            print(f"    {character_names[i]:30s} {scores[i]:+.2f}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation-dir",
        type=Path,
        required=True,
        help="Directory with character .pt files",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    # Load and analyze
    activation_matrix, character_names = load_character_activations(
        args.activation_dir, layer=args.layer
    )

    pca, transformed, scaler = run_pca(activation_matrix, args.n_components)
    print_pc_loadings(pca, transformed, character_names)

    # Save
    results = {
        "character_names": character_names,
        "activation_matrix": activation_matrix,
        "pca": pca,
        "transformed": transformed,
        "scaler": scaler,
        "layer": args.layer,
    }

    output_path = args.output_dir / "pca_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
