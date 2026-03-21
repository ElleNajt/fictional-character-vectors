#!/usr/bin/env python3
"""
Compute PC analysis for all universes and save structured results.

This does the heavy computation once so the notebook can just load and explore.

Outputs:
  results/pc_analysis.pkl - dict with all pre-computed analysis

Structure:
  {
    'universes': {
      'harry_potter': {
        'characters': ['Tom Riddle', ...],
        'char_names_full': ['harry_potter__tom_riddle', ...],
        'pc_scores': np.array (n_chars, n_pcs),
        'variance_explained': [0.12, 0.10, ...],
        'top_questions': {
          0: [(q_idx, corr, question_text), ...],  # PC1
          1: [...],  # PC2
        },
        'responses': {
          'harry_potter__tom_riddle': {
            0: "response to q0",
            1: "response to q1",
            ...
          },
          ...
        }
      },
      ...
    },
    'residualized': True/False,  # whether role space was projected out
    'model': 'qwen3-32b',
    'questions': [...],  # all 240 questions
  }
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler
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

    return np.array(question_activations) if question_activations else None


def load_responses(responses_dir: Path, char_name: str):
    """Load character responses to questions."""
    # Try .json first, then .jsonl
    json_file = responses_dir / f"{char_name}.json"
    jsonl_file = responses_dir / f"{char_name}.jsonl"

    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        # Format: {q_idx: response} or {q_idx: {variant: response}}
        responses = {}
        for k, v in data.items():
            if isinstance(v, dict):
                responses[int(k)] = list(v.values())[0] if v else ""
            elif isinstance(v, list):
                responses[int(k)] = v[0] if v else ""
            else:
                responses[int(k)] = v
        return responses

    if jsonl_file.exists():
        responses = {}
        with open(jsonl_file) as f:
            for line in f:
                item = json.loads(line)
                q_idx = item.get("question_id", item.get("q_idx"))
                resp = item.get("response", item.get("text", ""))
                if q_idx is not None:
                    responses[int(q_idx)] = resp
        return responses

    return None


def get_universe_indices(char_names: list, prefixes: list):
    """Get indices of characters belonging to a universe."""
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def compute_question_correlations(
    char_names: list,
    pc_scores: np.ndarray,
    pc_direction: np.ndarray,
    activations_dir: Path,
    role_scaler,
    role_pca,
    layer: int = 32,
):
    """
    Compute correlation between each question's projection and PC scores.

    Returns: list of (q_idx, correlation) sorted by |correlation|
    """
    # Load per-question activations for all characters
    char_question_projections = []
    valid_indices = []

    for i, cname in enumerate(char_names):
        acts = load_per_question_activations(activations_dir, cname, layer)
        if acts is None or len(acts) != 240:
            continue

        # Project through role space to get residuals
        acts_scaled = role_scaler.transform(acts)
        in_role_space = role_pca.transform(acts_scaled)
        reconstructed = in_role_space @ role_pca.components_
        residuals = acts_scaled - reconstructed

        # Project each question onto PC direction
        projections = residuals @ pc_direction
        char_question_projections.append(projections)
        valid_indices.append(i)

    if not char_question_projections:
        return []

    char_question_projections = np.array(char_question_projections)  # (n_chars, 240)
    valid_pc_scores = pc_scores[valid_indices]

    # Correlation of each question's projection with PC score
    correlations = []
    for q_idx in range(240):
        q_projections = char_question_projections[:, q_idx]
        if np.std(q_projections) > 0 and np.std(valid_pc_scores) > 0:
            corr = np.corrcoef(q_projections, valid_pc_scores)[0, 1]
        else:
            corr = 0
        correlations.append((q_idx, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: -abs(x[1]))
    return correlations


def analyze_universe(
    universe_key: str,
    prefixes: list,
    char_names: list,
    residuals: np.ndarray,
    activations_dir: Path,
    responses_dir: Path,
    questions: list,
    role_scaler,
    role_pca,
    n_pcs: int = 5,
    n_top_questions: int = 10,
    layer: int = 32,
    skip_correlations: bool = False,
):
    """Analyze a single universe and return structured results."""
    indices = get_universe_indices(char_names, prefixes)
    if len(indices) < 10:
        print(f"  {universe_key}: only {len(indices)} characters, skipping")
        return None

    u_names = [char_names[i] for i in indices]
    u_residuals = residuals[indices]

    # Fit PCA
    u_pca = SkPCA(n_components=min(n_pcs, len(indices) - 1))
    u_transformed = u_pca.fit_transform(u_residuals)

    # Get display names
    display_names = [n.split("__")[-1].replace("_", " ").title() for n in u_names]

    # Compute top questions for each PC (optional - slow)
    top_questions = {}

    if not skip_correlations:
        for pc_idx in range(u_pca.n_components_):
            pc_scores = u_transformed[:, pc_idx]
            pc_direction = u_pca.components_[pc_idx]

            correlations = compute_question_correlations(
                u_names,
                pc_scores,
                pc_direction,
                activations_dir,
                role_scaler,
                role_pca,
                layer,
            )

            # Store top questions with text
            top_questions[pc_idx] = [
                (q_idx, corr, questions[q_idx])
                for q_idx, corr in correlations[:n_top_questions]
            ]

    # Load responses for all characters (only for top questions across all PCs)
    all_top_q_indices = set()
    for pc_idx in top_questions:
        for q_idx, _, _ in top_questions[pc_idx]:
            all_top_q_indices.add(q_idx)

    char_responses = {}
    for cname in u_names:
        responses = load_responses(responses_dir, cname)
        if responses:
            # Only store responses to top questions
            char_responses[cname] = {
                q_idx: responses.get(q_idx, "[no response]")
                for q_idx in all_top_q_indices
                if q_idx in responses
            }

    return {
        "characters": display_names,
        "char_names_full": u_names,
        "pc_scores": u_transformed,
        "variance_explained": u_pca.explained_variance_ratio_.tolist(),
        "top_questions": top_questions,
        "responses": char_responses,
        "n_characters": len(u_names),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute PC analysis for all universes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/pc_analysis.pkl",
        help="Output pickle file",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3-32b", help="Model name for output metadata"
    )
    parser.add_argument("--layer", type=int, default=32, help="Layer for activations")
    parser.add_argument(
        "--n-pcs", type=int, default=5, help="Number of PCs to compute per universe"
    )
    parser.add_argument(
        "--n-top-questions", type=int, default=10, help="Number of top questions per PC"
    )
    parser.add_argument(
        "--skip-all", action="store_true", help="Skip the _all combined analysis (slow)"
    )
    parser.add_argument(
        "--skip-correlations",
        action="store_true",
        help="Skip question correlation analysis (very slow due to loading 750MB files)",
    )
    args = parser.parse_args()

    # Paths
    repo_root = Path(__file__).parent.parent
    results_pkl = repo_root / "results" / "fictional_character_analysis_filtered.pkl"
    role_pkl = repo_root / "data" / "role_vectors" / "qwen-3-32b_pca_layer32.pkl"
    questions_file = (
        repo_root / "assistant-axis" / "data" / "extraction_questions.jsonl"
    )
    activations_dir = (
        repo_root / "outputs" / "qwen3-32b_20260211_002840" / "activations"
    )
    responses_dir = repo_root / "outputs" / "qwen3-32b_20260211_002840" / "responses"

    print("Loading data...")

    # Load questions
    questions = []
    with open(questions_file) as f:
        for line in f:
            q = json.loads(line)
            questions.append(q["question"])
    print(f"  Loaded {len(questions)} questions")

    # Load character data
    with open(results_pkl, "rb") as f:
        results = pickle.load(f)
    char_names = results["character_names"]
    activation_matrix = results["activation_matrix"]
    print(f"  Loaded {len(char_names)} characters")

    # Load role vectors
    with open(role_pkl, "rb") as f:
        role_data = pickle.load(f)
    role_pca = role_data["pca"]
    role_scaler = role_data["scaler"]
    print(f"  Loaded {len(role_data['role_names'])} role vectors")

    # Compute residuals (after projecting out role space)
    print("Computing residuals...")
    chars_scaled = role_scaler.transform(activation_matrix)
    chars_in_role_space = role_pca.transform(chars_scaled)
    reconstructed = chars_in_role_space @ role_pca.components_
    residuals_role = chars_scaled - reconstructed

    # Analyze each universe
    print("\nAnalyzing universes...")
    universe_results = {}

    for universe_key, prefixes in tqdm(UNIVERSES.items(), desc="Universes"):
        result = analyze_universe(
            universe_key,
            prefixes,
            char_names,
            residuals_role,
            activations_dir,
            responses_dir,
            questions,
            role_scaler,
            role_pca,
            n_pcs=args.n_pcs,
            n_top_questions=args.n_top_questions,
            layer=args.layer,
            skip_correlations=args.skip_correlations,
        )
        if result:
            universe_results[universe_key] = result
            print(
                f"  {universe_key}: {result['n_characters']} chars, "
                f"PC1={result['variance_explained'][0]:.1%}"
            )

    # Also analyze ALL characters together (optional - slow)
    if not args.skip_all:
        print("\nAnalyzing all characters combined...")
        all_pca = SkPCA(n_components=args.n_pcs)
        all_transformed = all_pca.fit_transform(residuals_role)

        # Top questions for combined analysis
        all_top_questions = {}
        for pc_idx in range(all_pca.n_components_):
            pc_scores = all_transformed[:, pc_idx]
            pc_direction = all_pca.components_[pc_idx]

            correlations = compute_question_correlations(
                char_names,
                pc_scores,
                pc_direction,
                activations_dir,
                role_scaler,
                role_pca,
                args.layer,
            )

            all_top_questions[pc_idx] = [
                (q_idx, corr, questions[q_idx])
                for q_idx, corr in correlations[: args.n_top_questions]
            ]

        universe_results["_all"] = {
            "characters": [
                n.split("__")[-1].replace("_", " ").title() for n in char_names
            ],
            "char_names_full": char_names,
            "pc_scores": all_transformed,
            "variance_explained": all_pca.explained_variance_ratio_.tolist(),
            "top_questions": all_top_questions,
            "responses": {},  # Too many to store for all
            "n_characters": len(char_names),
        }
        print(
            f"  All: {len(char_names)} chars, PC1={all_pca.explained_variance_ratio_[0]:.1%}"
        )
    else:
        print("\nSkipping _all combined analysis (--skip-all)")

    # Save results
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "universes": universe_results,
        "residualized": True,
        "model": args.model,
        "layer": args.layer,
        "questions": questions,
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\nSaved to {output_path}")
    print(f"  {len(universe_results)} universes analyzed")


if __name__ == "__main__":
    main()
