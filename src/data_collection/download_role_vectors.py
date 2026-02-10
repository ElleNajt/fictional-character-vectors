#!/usr/bin/env python3
"""
Download role vectors from the Assistant Axis HuggingFace dataset
and fit a PCA for comparison with fictional characters.

The Assistant Axis paper (Lu et al., 2026) provides individual role vectors.
We download these and fit our own PCA to create a comparable space.

Usage:
    python src/data_collection/download_role_vectors.py --output data/role_vectors/
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def download_role_vectors(output_dir: Path, model: str = "qwen-3-32b"):
    """Download all role vectors for a model from HuggingFace."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # List all files in the dataset
    all_files = list_repo_files(
        "lu-christina/assistant-axis-vectors", repo_type="dataset"
    )

    # Filter to role vectors for our model
    role_files = [f for f in all_files if f.startswith(f"{model}/role_vectors/")]

    print(f"Found {len(role_files)} role vectors for {model}")

    vectors = {}
    for file_path in tqdm(role_files, desc="Downloading"):
        role_name = Path(file_path).stem

        local_path = hf_hub_download(
            repo_id="lu-christina/assistant-axis-vectors",
            filename=file_path,
            repo_type="dataset",
        )

        data = torch.load(local_path, map_location="cpu", weights_only=False)

        # Extract the vector - format may be tensor or dict with 'vector' key
        if isinstance(data, dict) and "vector" in data:
            vector = data["vector"]
        elif isinstance(data, torch.Tensor):
            vector = data
        else:
            print(f"  Warning: unexpected format for {role_name}")
            continue

        vectors[role_name] = vector

    return vectors


def fit_pca_on_roles(vectors: dict, layer: int = 32, n_components: int = None):
    """
    Fit PCA on role vectors at a specific layer.

    Args:
        vectors: Dict mapping role names to tensors of shape (n_layers, hidden_dim)
        layer: Which layer to use (default 32 for Qwen 2.5 32B)
        n_components: Number of PCA components (default: min of n_roles, hidden_dim)

    Returns:
        pca: Fitted PCA model
        scaler: Fitted StandardScaler
        role_names: List of role names in order
        transformed: PCA-transformed role vectors
    """

    role_names = sorted(vectors.keys())

    # Extract vectors at specified layer
    layer_vectors = []
    for role in role_names:
        vec = vectors[role]
        if vec.dim() == 2:  # (n_layers, hidden_dim)
            layer_vec = vec[layer].float().numpy()
        elif vec.dim() == 1:  # (hidden_dim,) - already single layer
            layer_vec = vec.float().numpy()
        else:
            print(f"Warning: unexpected shape {vec.shape} for {role}")
            continue
        layer_vectors.append(layer_vec)

    # Stack into matrix
    X = np.vstack(layer_vectors)
    print(f"Role matrix shape: {X.shape}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X_scaled)

    print(f"PCA fitted: {pca.n_components_} components")
    print(f"Variance explained: {sum(pca.explained_variance_ratio_):.1%}")

    return pca, scaler, role_names, transformed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/role_vectors"))
    parser.add_argument("--model", type=str, default="qwen-3-32b")
    parser.add_argument("--layer", type=int, default=32)
    args = parser.parse_args()

    print(f"Downloading role vectors for {args.model}...")
    vectors = download_role_vectors(args.output, args.model)

    print(f"\nFitting PCA on layer {args.layer}...")
    pca, scaler, role_names, transformed = fit_pca_on_roles(vectors, args.layer)

    # Save everything
    output_file = args.output / f"{args.model}_pca_layer{args.layer}.pkl"
    results = {
        "pca": pca,
        "scaler": scaler,
        "role_names": role_names,
        "transformed": transformed,
        "layer": args.layer,
        "model": args.model,
        "variance_explained": pca.explained_variance_ratio_,
    }

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved to {output_file}")
    print(f"  {len(role_names)} roles")
    print(f"  {pca.n_components_} PCA components")
    print(f"  Layer {args.layer}")


if __name__ == "__main__":
    main()
