"""Regress residual PC scores on LLM-coded features.

Loads llm_feature_coded.json and computes correlations + regression R²
for each universe × PC. Saves results to results/feature_regression.json
for use in blogpost.org.

Usage:
    python src/analysis/feature_regression.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA

CHAR_DATA_PATH = "results/fictional_character_analysis_filtered.pkl"
LU_PCA_PATH = "data/role_vectors/qwen-3-32b_pca_layer32.pkl"
CODED_PATH = "results/llm_feature_coded.json"
OUTPUT_PATH = "results/feature_regression.json"

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


def main():
    with open(CHAR_DATA_PATH, "rb") as f:
        char_data = pickle.load(f)
    with open(LU_PCA_PATH, "rb") as f:
        lu_data = pickle.load(f)
    with open(CODED_PATH) as f:
        coded = json.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = lu_data["pca"]
    scaler = lu_data["scaler"]

    chars_scaled = scaler.transform(activation_matrix)
    chars_in_role_space = role_pca.transform(chars_scaled)
    reconstructed = chars_in_role_space @ role_pca.components_
    residuals = chars_scaled - reconstructed

    results = {}

    for schema_key, data in coded.items():
        schema = data["schema"]
        characters = data["characters"]
        universe = schema["universe"]
        pc_num = schema["pc"]
        features = schema["features"]
        feature_names = [f["name"] for f in features]

        prefixes = ALL_UNIVERSES[universe]
        indices = get_universe_indices(char_names, prefixes)
        u_names = [char_names[i] for i in indices]
        u_residuals = residuals[indices]

        # Fit PCA on universe residuals
        u_pca = SkPCA(n_components=max(2, pc_num))
        u_scores = u_pca.fit_transform(u_residuals)
        pc_scores = u_scores[:, pc_num - 1]

        # Build feature matrix (only chars with valid ratings)
        valid_indices = []
        feature_matrix = []
        for i, cname in enumerate(u_names):
            if cname not in characters:
                continue
            ratings = characters[cname]
            if "error" in ratings:
                continue
            row = []
            valid = True
            for fname in feature_names:
                val = ratings.get(fname)
                if val is None or not isinstance(val, (int, float)):
                    valid = False
                    break
                row.append(float(val))
            if valid:
                valid_indices.append(i)
                feature_matrix.append(row)

        if len(valid_indices) < 10:
            print(f"{schema_key}: too few valid chars ({len(valid_indices)})")
            continue

        X = np.array(feature_matrix)
        y = pc_scores[valid_indices]
        n = len(y)

        # Per-feature correlations
        correlations = []
        for j, fname in enumerate(feature_names):
            r = np.corrcoef(X[:, j], y)[0, 1]
            correlations.append(
                {
                    "feature": fname,
                    "correlation": float(r),
                    "abs_correlation": float(abs(r)),
                }
            )
        correlations.sort(key=lambda x: -x["abs_correlation"])

        # Multiple regression R²
        X_with_intercept = np.column_stack([np.ones(n), X])
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            y_hat = X_with_intercept @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot
            # Adjusted R²
            p = X.shape[1]
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        except np.linalg.LinAlgError:
            r_squared = None
            adj_r_squared = None
            beta = None

        # Feature coefficients (standardized)
        if beta is not None:
            x_std = X.std(axis=0)
            y_std = y.std()
            std_beta = beta[1:] * x_std / y_std
            coefs = [
                {"feature": fname, "std_beta": float(std_beta[j])}
                for j, fname in enumerate(feature_names)
            ]
        else:
            coefs = []

        results[schema_key] = {
            "universe": universe,
            "pc": pc_num,
            "n_chars": n,
            "n_features": len(feature_names),
            "features": feature_names,
            "correlations": correlations,
            "r_squared": float(r_squared) if r_squared is not None else None,
            "adj_r_squared": float(adj_r_squared)
            if adj_r_squared is not None
            else None,
            "coefficients": coefs,
            "high_chars": schema["high_chars"],
            "low_chars": schema["low_chars"],
        }

        print(f"{schema_key} (n={n}): R²={r_squared:.3f}, adj R²={adj_r_squared:.3f}")
        for c in correlations[:3]:
            print(f"  {c['feature']}: r={c['correlation']:+.3f}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
