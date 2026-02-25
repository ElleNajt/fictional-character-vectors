#!/usr/bin/env python3
"""
Pre-compute per-question projections onto PC directions.

This loads each character's activation file once and projects all 240 questions
onto each universe's PC directions. The result is much smaller and can be used
for fast correlation analysis.

Output: results/question_projections.pkl
  {
    'harry_potter': {
      'char_names': [...],
      'projections': np.array (n_chars, n_questions=240, n_pcs=5),
      'pc_directions': np.array (n_pcs, hidden_dim),
    },
    ...
  }
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA as SkPCA
from tqdm import tqdm

UNIVERSES = {
    "harry_potter": ["harry_potter__", "harry_potter_series__"],
    "star_wars": ["star_wars__"],
    "marvel": ["marvel__", "marvel_comics__"],
    "game_of_thrones": ["game_of_thrones__", "a_song_of_ice_and_fire__"],
    "lord_of_the_rings": ["lord_of_the_rings__", "the_lord_of_the_rings__"],
    "naruto": ["naruto__"],
    "greek_mythology": ["greek_mythology__"],
    "chinese_mythology": [
        "chinese_mythology__",
        "journey_to_the_west__",
        "romance_of_the_three_kingdoms__",
    ],
    "hindu_mythology": [
        "hindu_mythology__",
        "hindu_buddhist_mythology__",
        "mahabharata__",
        "ramayana__",
    ],
    "norse_mythology": ["norse_mythology__"],
    "egyptian_mythology": ["egyptian_mythology__"],
    "shakespeare": [
        "shakespeare__",
        "hamlet__",
        "macbeth__",
        "othello__",
        "king_lear__",
        "romeo_and_juliet__",
        "a_midsummer_nights_dream__",
        "much_ado_about_nothing__",
        "as_you_like_it__",
        "henry_iv__",
        "henry_v__",
        "richard_iii__",
    ],
}


def get_universe_indices(char_names: list, prefixes: list):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def load_per_question_activations(
    activations_dir: Path, char_name: str, layer: int = 32
):
    """Load per-question activations for a character."""
    pt_file = activations_dir / f"{char_name}.pt"
    if not pt_file.exists():
        return None

    data = torch.load(pt_file, map_location="cpu", weights_only=False)
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
                    q_acts.append(act[layer - 1] if act.shape[0] >= layer else act[-1])
                else:
                    q_acts.append(act)
        if q_acts:
            question_activations.append(np.mean(q_acts, axis=0))

    return np.array(question_activations) if len(question_activations) == 240 else None


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute question projections onto PCs"
    )
    parser.add_argument(
        "--output", type=str, default="results/question_projections.pkl"
    )
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--n-pcs", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    results_pkl = repo_root / "results" / "fictional_character_analysis_filtered.pkl"
    role_pkl = repo_root / "data" / "role_vectors" / "qwen-3-32b_pca_layer32.pkl"
    activations_dir = (
        repo_root / "outputs" / "qwen3-32b_20260211_002840" / "activations"
    )

    print("Loading data...")
    with open(results_pkl, "rb") as f:
        results = pickle.load(f)
    char_names = results["character_names"]
    activation_matrix = results["activation_matrix"]

    with open(role_pkl, "rb") as f:
        role_data = pickle.load(f)
    role_pca = role_data["pca"]

    # Compute residuals
    print("Computing residuals...")
    chars_centered = activation_matrix - role_pca.mean_
    chars_in_role_space = chars_centered @ role_pca.components_.T
    reconstructed = chars_in_role_space @ role_pca.components_
    residuals_role = chars_centered - reconstructed

    output_data = {}

    for universe_key, prefixes in UNIVERSES.items():
        print(f"\n=== {universe_key} ===")

        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 10:
            print(f"  Only {len(indices)} chars, skipping")
            continue

        u_names = [char_names[i] for i in indices]
        u_residuals = residuals_role[indices]

        # Fit PCA to get PC directions
        u_pca = SkPCA(n_components=args.n_pcs)
        u_pca.fit(u_residuals)
        pc_directions = u_pca.components_  # (n_pcs, hidden_dim)

        # For each character, load activations and project onto PCs
        projections = []  # Will be (n_chars, 240, n_pcs)
        valid_names = []

        for cname in tqdm(u_names, desc=f"  Loading {universe_key}"):
            acts = load_per_question_activations(activations_dir, cname, args.layer)
            if acts is None:
                continue

            # Get residuals for each question
            acts_centered = acts - role_pca.mean_
            in_role = acts_centered @ role_pca.components_.T
            recon = in_role @ role_pca.components_
            q_residuals = acts_centered - recon  # (240, hidden_dim)

            # Project onto each PC direction
            char_proj = q_residuals @ pc_directions.T  # (240, n_pcs)
            projections.append(char_proj)
            valid_names.append(cname)

        if projections:
            output_data[universe_key] = {
                "char_names": valid_names,
                "projections": np.array(projections),  # (n_chars, 240, n_pcs)
                "pc_directions": pc_directions,
                "variance_explained": u_pca.explained_variance_ratio_.tolist(),
            }
            print(f"  {len(valid_names)}/{len(u_names)} chars with activations")

    # Save
    output_path = repo_root / args.output
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
