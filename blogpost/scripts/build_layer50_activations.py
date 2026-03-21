#!/usr/bin/env python3
"""
Build layer-50 activation matrices for characters and roles.

Output:
  results/layer50_activations.pkl      {character_names, activation_matrix, layer}
  results/roles_layer50_activations.pkl {role_names, activation_matrix, layer}

Run from repo root:
    python blogpost/scripts/build_layer50_activations.py
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

LAYER = 50


def build_character_matrix(vectors_dir, layer):
    names = []
    vectors = []
    seen = set()
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "vector" in data:
            vec = data["vector"][layer]
        elif isinstance(data, dict):
            vec = data.get(layer, data.get(str(layer), list(data.values())[0]))
        else:
            vec = data
        if isinstance(vec, torch.Tensor):
            vec = vec.float().numpy()
        vec = vec.flatten()

        # Deduplicate by short name (part after __)
        short = pt_file.stem.split("__")[-1] if "__" in pt_file.stem else pt_file.stem
        if short in seen:
            continue
        seen.add(short)

        names.append(pt_file.stem)
        vectors.append(vec)

    return names, np.array(vectors)


def build_role_matrix(layer, model="qwen-3-32b"):
    all_files = list_repo_files("lu-christina/assistant-axis-vectors", repo_type="dataset")
    role_files = sorted(f for f in all_files if f.startswith(f"{model}/role_vectors/"))
    print(f"Found {len(role_files)} role vectors")

    names = []
    vectors = []
    for file_path in tqdm(role_files, desc="Downloading roles"):
        local_path = hf_hub_download(
            repo_id="lu-christina/assistant-axis-vectors",
            filename=file_path,
            repo_type="dataset",
        )
        data = torch.load(local_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "vector" in data:
            vec = data["vector"]
        elif isinstance(data, torch.Tensor):
            vec = data
        else:
            continue

        if vec.dim() == 2:
            vec = vec[layer].float().numpy()
        else:
            vec = vec.float().numpy()

        names.append(Path(file_path).stem)
        vectors.append(vec.flatten())

    return names, np.array(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir", default="outputs/qwen3-32b_20260211_002840/vectors")
    parser.add_argument("--layer", type=int, default=LAYER)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    vectors_dir = repo_root / args.vectors_dir
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Characters
    print(f"Building character matrix at layer {args.layer}...")
    char_names, char_matrix = build_character_matrix(vectors_dir, args.layer)
    print(f"  {len(char_names)} characters, shape {char_matrix.shape}")

    char_path = results_dir / "layer50_activations.pkl"
    with open(char_path, "wb") as f:
        pickle.dump({"character_names": char_names, "activation_matrix": char_matrix, "layer": args.layer}, f)
    print(f"  Saved to {char_path}")

    # Roles
    print(f"\nBuilding role matrix at layer {args.layer}...")
    role_names, role_matrix = build_role_matrix(args.layer)
    print(f"  {len(role_names)} roles, shape {role_matrix.shape}")

    role_path = results_dir / "roles_layer50_activations.pkl"
    with open(role_path, "wb") as f:
        pickle.dump({"role_names": role_names, "activation_matrix": role_matrix, "layer": args.layer}, f)
    print(f"  Saved to {role_path}")


if __name__ == "__main__":
    main()
