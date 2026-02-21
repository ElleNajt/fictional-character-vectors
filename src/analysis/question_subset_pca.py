"""
Question subset PCA comparison.

Tests whether the top-10 most informative questions produce the same PC1
as all 240 questions. Computes per-question variance along residual PC1,
picks the top-10 and bottom-10, and compares PC1 rankings.

Output: results/question_subset_pca.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA as SkPCA

BASE = Path("/workspace-vast/lnajt/persona_vectors/fictional-character-vectors")
ACTIVATIONS_DIR = BASE / "outputs/qwen3-32b_20260211_002840/activations"
RESULTS_DIR = BASE / "results"

ALL_UNIVERSES = {
    "Harry Potter": ["harry_potter__", "harry_potter_series__"],
    "Star Wars": ["star_wars__"],
    "LOTR": ["lord_of_the_rings__"],
    "Marvel": ["marvel__", "marvel_comics__"],
    "Game of Thrones": ["game_of_thrones__"],
    "Naruto": ["naruto__"],
    "Greek Mythology": ["greek_mythology__"],
    "Chinese Mythology": ["chinese_mythology__"],
    "Hindu Mythology": ["hindu_mythology__"],
    "Norse Mythology": ["norse_mythology__"],
    "Egyptian Mythology": ["egyptian_mythology__"],
    "Shakespeare": ["shakespeare__"],
}


def get_universe_indices(char_names, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def load_per_question_activations(char_name, layer=32):
    """Load per-question mean activations for a character."""
    pt_file = ACTIVATIONS_DIR / f"{char_name}.pt"
    if not pt_file.exists():
        return None
    data = torch.load(pt_file, map_location="cpu", weights_only=True)
    question_activations = {}
    for q_idx in range(240):
        positions = []
        for p_idx in range(5):
            key = f"pos_p{p_idx}_q{q_idx}"
            if key in data:
                act = data[key].float()  # (64, 5120)
                positions.append(act[layer - 1])  # layer 32 = index 31
        if positions:
            question_activations[q_idx] = torch.stack(positions).mean(dim=0).numpy()
    return question_activations


def main():
    print("=== Question Subset PCA Comparison ===\n")

    with open(RESULTS_DIR / "fictional_character_analysis_filtered.pkl", "rb") as f:
        char_data = pickle.load(f)
    with open(str(BASE / "data/role_vectors/qwen-3-32b_pca_layer32.pkl"), "rb") as f:
        role_data = pickle.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = role_data["pca"]
    role_scaler = role_data["scaler"]

    results = {}

    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            continue

        u_names = [char_names[i] for i in indices]
        print(f"\n{universe}: loading {len(u_names)} characters...")

        # Load per-question activations for all characters in universe
        per_q_acts = {}
        for name in u_names:
            acts = load_per_question_activations(name)
            if acts and len(acts) == 240:
                per_q_acts[name] = acts
            if len(per_q_acts) % 20 == 0 and len(per_q_acts) > 0:
                print(f"  loaded {len(per_q_acts)} characters...")

        names_with_acts = [n for n in u_names if n in per_q_acts]
        print(
            f"  {len(names_with_acts)}/{len(u_names)} characters have full activations"
        )

        if len(names_with_acts) < 20:
            print(f"  skipping (need >= 20)")
            continue

        # Step 1: Compute all-240 mean vectors and do residual PCA to get PC1 direction
        all_vectors = []
        for name in names_with_acts:
            mean_vec = np.mean([per_q_acts[name][q] for q in range(240)], axis=0)
            all_vectors.append(mean_vec)
        all_vectors = np.array(all_vectors)

        all_scaled = role_scaler.transform(all_vectors)
        all_in_role = all_scaled @ role_pca.components_.T @ role_pca.components_
        all_residuals = all_scaled - all_in_role

        all_pca = SkPCA(n_components=1)
        all_scores = all_pca.fit_transform(all_residuals).flatten()
        pc1_dir = all_pca.components_[0]

        # Step 2: Compute per-question variance along residual PC1
        # For each question, compute each character's projection onto PC1
        q_variances = []
        for q_idx in range(240):
            q_vectors = []
            for name in names_with_acts:
                q_vectors.append(per_q_acts[name][q_idx])
            q_vectors = np.array(q_vectors)
            q_scaled = role_scaler.transform(q_vectors)
            q_in_role = q_scaled @ role_pca.components_.T @ role_pca.components_
            q_residuals = q_scaled - q_in_role
            q_projections = q_residuals @ pc1_dir
            q_variances.append(float(np.var(q_projections)))

        sorted_by_var = np.argsort(q_variances)[::-1]
        top10 = sorted_by_var[:10].tolist()
        top20 = sorted_by_var[:20].tolist()
        top50 = sorted_by_var[:50].tolist()
        bottom10 = sorted_by_var[-10:].tolist()

        np.random.seed(42)
        rand10 = np.random.choice(240, 10, replace=False).tolist()
        rand20 = np.random.choice(240, 20, replace=False).tolist()

        # Step 3: For each subset, compute mean vectors and residual PCA scores
        def scores_from_subset(q_indices):
            vectors = []
            for name in names_with_acts:
                vecs = [per_q_acts[name][q] for q in q_indices if q in per_q_acts[name]]
                if vecs:
                    vectors.append(np.mean(vecs, axis=0))
                else:
                    return None
            X = np.array(vectors)
            X_scaled = role_scaler.transform(X)
            X_in_role = X_scaled @ role_pca.components_.T @ role_pca.components_
            X_resid = X_scaled - X_in_role
            pca = SkPCA(n_components=1)
            scores = pca.fit_transform(X_resid).flatten()
            return scores

        subsets = {
            "all_240": list(range(240)),
            "top_10": top10,
            "top_20": top20,
            "top_50": top50,
            "bottom_10": bottom10,
            "random_10": rand10,
            "random_20": rand20,
        }

        subset_scores = {}
        for label, q_indices in subsets.items():
            scores = scores_from_subset(q_indices)
            if scores is not None:
                subset_scores[label] = scores

        # Step 4: Ranking correlations (absolute value since PCA sign is arbitrary)
        correlations = {}
        ref = subset_scores.get("all_240")
        if ref is not None:
            for label, scores in subset_scores.items():
                if label == "all_240":
                    continue
                r = abs(np.corrcoef(ref, scores)[0, 1])
                correlations[label] = float(r)

        results[universe] = {
            "n_chars": len(names_with_acts),
            "correlations": correlations,
            "top10_q_indices": top10,
            "q_variances_top5": [float(q_variances[i]) for i in sorted_by_var[:5]],
            "q_variances_bottom5": [float(q_variances[i]) for i in sorted_by_var[-5:]],
        }

        corr_strs = [f"{k}={v:.3f}" for k, v in sorted(correlations.items())]
        print(f"  Correlations with all-240: {', '.join(corr_strs)}")

    # Save
    output_path = RESULTS_DIR / "question_subset_pca.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
