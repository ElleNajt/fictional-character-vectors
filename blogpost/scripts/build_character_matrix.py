#!/usr/bin/env python3
"""
Build the character activation matrix from per-character vector files.

Input: outputs/.../vectors/*.pt  (one per character, each containing mean activation)
Output: results/fictional_character_analysis_filtered.pkl
  {
    'character_names': [...],
    'activation_matrix': np.array (n_chars, hidden_dim),
  }

Run from repo root:
    python blogpost/scripts/build_character_matrix.py
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def deduplicate_characters(names, vectors):
    """Keep first occurrence of each character name (part after __)."""
    seen = set()
    keep_indices = []
    for i, name in enumerate(names):
        char_name = name.split("__")[-1] if "__" in name else name
        if char_name not in seen:
            seen.add(char_name)
            keep_indices.append(i)

    return [names[i] for i in keep_indices], [vectors[i] for i in keep_indices]


def main():
    parser = argparse.ArgumentParser(
        description="Build character activation matrix from vector files"
    )
    parser.add_argument(
        "--vectors-dir",
        type=str,
        default="outputs/qwen3-32b_20260211_002840/vectors",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/fictional_character_analysis_filtered.pkl",
    )
    parser.add_argument("--layer", type=int, default=32)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    vectors_dir = repo_root / args.vectors_dir
    output_path = repo_root / args.output

    if not vectors_dir.exists():
        print(f"ERROR: vectors directory not found: {vectors_dir}")
        print("Run extraction first. See README.org")
        return

    char_names = []
    vectors = []
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "vector" in data:
            vec = data["vector"][args.layer]
        elif isinstance(data, dict):
            vec = data.get(
                args.layer, data.get(str(args.layer), list(data.values())[0])
            )
        else:
            vec = data
        if isinstance(vec, torch.Tensor):
            vec = vec.float().numpy()
        vectors.append(vec.flatten())
        char_names.append(pt_file.stem)

    n_before = len(char_names)
    char_names, vectors = deduplicate_characters(char_names, vectors)
    if n_before != len(char_names):
        print(f"Deduplicated: {n_before} -> {len(char_names)} characters")

    activation_matrix = np.array(vectors)
    print(f"Loaded {len(char_names)} character vectors from {vectors_dir}")
    print(f"Activation matrix shape: {activation_matrix.shape}")

    results = {
        "character_names": char_names,
        "activation_matrix": activation_matrix,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
