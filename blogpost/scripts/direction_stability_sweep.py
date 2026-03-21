#!/usr/bin/env python3
"""
Direction stability sweep: how many questions are needed to recover PC directions?

For each subset size k, draws random subsets of k questions, averages
per-question activations in full 5120-dim space, fits PCA, and compares
recovered PC directions to full-battery PCA directions via cosine similarity.

Output: results/direction_stability_sweep.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA as SkPCA

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

LAYER = 32
N_QUESTIONS = 240
N_PROMPTS = 5
N_PCS = 5
N_DRAWS = 200
SUBSET_SIZES = [1, 2, 3, 5, 10, 15, 20, 30, 50, 80, 120, 240]


def load_per_question_activations(pt_file: Path) -> np.ndarray:
    """Load per-question activations for a character. Returns (240, 5120)."""
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    question_acts = []
    for q_idx in range(N_QUESTIONS):
        prompt_acts = []
        for p_idx in range(N_PROMPTS):
            key = f"pos_p{p_idx}_q{q_idx}"
            if key in data:
                act = data[key]  # (64, 5120)
                prompt_acts.append(act[LAYER - 1].float().numpy())
        if prompt_acts:
            question_acts.append(np.mean(prompt_acts, axis=0))
    if len(question_acts) == N_QUESTIONS:
        return np.array(question_acts)  # (240, 5120)
    return None


def get_universe_indices(char_names, prefixes):
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_pkl = repo_root / "results" / "fictional_character_analysis_filtered.pkl"
    role_pkl = repo_root / "data" / "role_vectors" / "qwen-3-32b_pca_layer32.pkl"
    activations_dir = (
        repo_root / "outputs" / "qwen3-32b_20260211_002840" / "activations"
    )

    print("Loading metadata...")
    with open(results_pkl, "rb") as f:
        results = pickle.load(f)
    char_names = results["character_names"]
    activation_matrix = results[
        "activation_matrix"
    ]  # (n_chars, 5120) full-battery mean

    with open(role_pkl, "rb") as f:
        role_data = pickle.load(f)
    role_mean = role_data["pca"].mean_

    rng = np.random.default_rng(42)
    output = {}

    for universe_key, prefixes in UNIVERSES.items():
        print(f"\n=== {universe_key} ===")
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            print(f"  Only {len(indices)} chars, skipping")
            continue

        u_names = [char_names[i] for i in indices]

        # Load per-question activations for this universe
        print(f"  Loading {len(u_names)} characters...")
        per_q_acts = []  # (n_chars, 240, 5120)
        valid_indices = []
        for ci, cname in enumerate(u_names):
            pt_file = activations_dir / f"{cname}.pt"
            if not pt_file.exists():
                continue
            acts = load_per_question_activations(pt_file)
            if acts is not None:
                per_q_acts.append(acts)
                valid_indices.append(ci)
        per_q_acts = np.array(per_q_acts)  # (n_valid, 240, 5120)
        n_valid = len(per_q_acts)
        print(f"  {n_valid}/{len(u_names)} chars loaded")

        # Full-battery PCA (mean over all 240 questions, centered by role mean)
        full_mean = per_q_acts.mean(axis=1) - role_mean  # (n_valid, 5120)
        pca_full = SkPCA(n_components=N_PCS).fit(full_mean)

        # Direction stability sweep
        pc_sweeps = {}
        for pc_i in range(N_PCS):
            pc_sweeps[f"PC{pc_i + 1}"] = []

        for size in SUBSET_SIZES:
            print(f"  size={size}", end="", flush=True)
            cosines_per_draw = {pc_i: [] for pc_i in range(N_PCS)}
            for draw in range(N_DRAWS):
                q_idx = rng.choice(N_QUESTIONS, size=size, replace=False)
                subset_mean = (
                    per_q_acts[:, q_idx, :].mean(axis=1) - role_mean
                )  # (n_valid, 5120)
                pca_sub = SkPCA(n_components=N_PCS).fit(subset_mean)
                for pc_i in range(N_PCS):
                    cos = abs(
                        np.dot(pca_full.components_[pc_i], pca_sub.components_[pc_i])
                    )
                    cosines_per_draw[pc_i].append(cos)

            for pc_i in range(N_PCS):
                vals = cosines_per_draw[pc_i]
                pc_sweeps[f"PC{pc_i + 1}"].append(
                    {
                        "size": size,
                        "median": float(np.median(vals)),
                        "p5": float(np.percentile(vals, 5)),
                        "p25": float(np.percentile(vals, 25)),
                        "p75": float(np.percentile(vals, 75)),
                        "p95": float(np.percentile(vals, 95)),
                    }
                )
            print(f" done (PC1 median={np.median(cosines_per_draw[0]):.3f})")

        output[universe_key] = {
            "n_chars": n_valid,
            "n_pcs": N_PCS,
            "sweeps": pc_sweeps,
        }

    # Save
    out_path = repo_root / "results" / "direction_stability_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
