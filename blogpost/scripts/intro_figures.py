"""
Generate intro figures for the blog post.

1. Eigenspectrum: cumulative variance for fiction vs roles
2. 2D scatter: characters on within-universe PC1 vs PC2, colored by universe
3. k=1 principal angle heatmap (standalone, larger)
4. AA interpretation: scatter of characters on formality vs warmth, colored by AA score

Output: results/intro_eigenspectrum.png
        results/intro_scatter_pc12.png
        results/intro_heatmap_k1.png
        results/intro_aa_features.png
"""

import json
import pickle

import matplotlib
import numpy as np
import torch
from scipy.linalg import svd
from sklearn.decomposition import PCA as SkPCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALL_UNIVERSES = {
    "Harry Potter": ["harry_potter__", "harry_potter_series__"],
    "Star Wars": ["star_wars__"],
    "LOTR": ["lord_of_the_rings__"],
    "Marvel": ["marvel__", "marvel_comics__"],
    "Game of Thrones": ["game_of_thrones__"],
    "Naruto": ["naruto__"],
    "Greek Myth.": ["greek_mythology__"],
    "Chinese Myth.": ["chinese_mythology__"],
    "Hindu Myth.": ["hindu_mythology__"],
    "Norse Myth.": ["norse_mythology__"],
    "Egyptian Myth.": ["egyptian_mythology__"],
    "Shakespeare": ["shakespeare__"],
}

# Distinct colors for 12 universes
UNIVERSE_COLORS = {
    "Harry Potter": "#e41a1c",
    "Star Wars": "#377eb8",
    "LOTR": "#4daf4a",
    "Marvel": "#984ea3",
    "Game of Thrones": "#ff7f00",
    "Naruto": "#a65628",
    "Greek Myth.": "#f781bf",
    "Chinese Myth.": "#999999",
    "Hindu Myth.": "#e7298a",
    "Norse Myth.": "#66a61e",
    "Egyptian Myth.": "#e6ab02",
    "Shakespeare": "#1b9e77",
}


def get_universe_indices(char_names, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i for i, name in enumerate(char_names) if any(name.startswith(p) for p in prefixes)
    ]


def principal_angles_cosines(A, B):
    M = A @ B.T
    s = svd(M, compute_uv=False)
    return np.clip(s, 0, 1)


def make_eigenspectrum(chars_centered, role_pca):
    """Fig 1: cumulative variance explained, fiction vs roles."""
    pca_fiction = SkPCA(n_components=200).fit(chars_centered)
    fvar = pca_fiction.explained_variance_ratio_
    fcumvar = np.cumsum(fvar)
    lu_var = role_pca.explained_variance_ratio_
    lu_cumvar = np.cumsum(lu_var)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(fcumvar) + 1), fcumvar, label=f"Fiction (1,268 characters)", color="#377eb8", linewidth=2)
    ax.plot(range(1, len(lu_cumvar) + 1), lu_cumvar, label=f"Roles (275, Lu et al.)", color="#e41a1c", linewidth=2)

    # Mark key thresholds
    for thresh, style in [(0.7, "--"), (0.9, ":")]:
        ax.axhline(thresh, color="gray", linestyle=style, alpha=0.5, linewidth=0.8)
        n_f = int(np.searchsorted(fcumvar, thresh) + 1)
        n_l = int(np.searchsorted(lu_cumvar, thresh) + 1)
        ax.annotate(f"{thresh:.0%}", xy=(195, thresh), fontsize=8, color="gray", va="bottom")

    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Both spaces are low-dimensional")
    ax.set_xlim(1, 200)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/intro_eigenspectrum.png", dpi=150, bbox_inches="tight")
    print("Saved results/intro_eigenspectrum.png")
    return pca_fiction


def make_scatter(chars_centered, char_names, pca_fiction):
    """Fig 2: 2D scatter on global fiction PC1 vs PC2, colored by universe."""
    scores = pca_fiction.transform(chars_centered)
    evr = pca_fiction.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ungrouped characters first (faint)
    grouped_set = set()
    for prefixes in ALL_UNIVERSES.values():
        for i, name in enumerate(char_names):
            if any(name.startswith(p) for p in prefixes):
                grouped_set.add(i)
    ungrouped = [i for i in range(len(char_names)) if i not in grouped_set]
    ax.scatter(scores[ungrouped, 0], scores[ungrouped, 1], c="#cccccc", s=8, alpha=0.3, label="Other", zorder=1)

    # Plot each universe
    for name, prefixes in ALL_UNIVERSES.items():
        idx = get_universe_indices(char_names, prefixes)
        ax.scatter(
            scores[idx, 0], scores[idx, 1],
            c=UNIVERSE_COLORS[name], s=12, alpha=0.6, label=name, zorder=2,
        )

    ax.set_xlabel(f"Fiction PC1 ({evr[0]:.0%} var)")
    ax.set_ylabel(f"Fiction PC2 ({evr[1]:.0%} var)")
    ax.set_title("1,268 fictional characters in activation space")
    ax.legend(loc="upper left", fontsize=7, ncol=2, markerscale=2)
    fig.tight_layout()
    fig.savefig("results/intro_scatter_pc12.png", dpi=150, bbox_inches="tight")
    print("Saved results/intro_scatter_pc12.png")


def make_heatmap_k1(chars_centered, char_names, role_pca, aa_norm):
    """Fig 3: standalone k=1 principal angle heatmap."""
    universe_pcas = {}
    for name, prefixes in ALL_UNIVERSES.items():
        idx = get_universe_indices(char_names, prefixes)
        n_comp = min(len(idx) - 1, 10)
        pca = SkPCA(n_components=n_comp).fit(chars_centered[idx])
        universe_pcas[name] = pca.components_

    labels = list(ALL_UNIVERSES.keys()) + ["Role PCs", "Asst. Axis"]
    n = len(labels)
    mat = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            if labels[i] == "Asst. Axis":
                basis_i = aa_norm.reshape(1, -1)
            elif labels[i] == "Role PCs":
                basis_i = role_pca.components_[:1]
            else:
                basis_i = universe_pcas[labels[i]][:1]

            if labels[j] == "Asst. Axis":
                basis_j = aa_norm.reshape(1, -1)
            elif labels[j] == "Role PCs":
                basis_j = role_pca.components_[:1]
            else:
                basis_j = universe_pcas[labels[j]][:1]

            cosines = principal_angles_cosines(basis_i, basis_j)
            mat[i, j] = cosines[0]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]) and i != j:
                color = "white" if mat[i, j] < 0.4 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("PC1 alignment across universes\n(cosine between within-universe PC1 directions)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig("results/intro_heatmap_k1.png", dpi=150, bbox_inches="tight")
    print("Saved results/intro_heatmap_k1.png")


def make_aa_figure(chars_centered, char_names, aa_norm):
    """Fig 4: AA feature interpretation.

    For each universe, plot adj R² from the feature regression, showing
    that the AA is predictable from LLM-discovered features.
    Also show which feature categories dominate (prosociality vs aggression).
    """
    with open("results/feature_regression_aa.json") as f:
        aa_reg = json.load(f)

    # Bar chart: adj R² per universe, colored by franchise vs mythology
    franchises = {"Harry Potter", "Star Wars", "LOTR", "Marvel", "Game of Thrones", "Naruto"}

    universes = []
    r2s = []
    colors = []
    for key, entry in aa_reg.items():
        u = entry["universe"]
        universes.append(u)
        r2s.append(entry["adj_r_squared"])
        colors.append("#377eb8" if u in franchises else "#e41a1c")

    # Sort by R²
    order = np.argsort(r2s)[::-1]
    universes = [universes[i] for i in order]
    r2s = [r2s[i] for i in order]
    colors = [colors[i] for i in order]

    # Also gather top features across universes
    feature_counts = {}
    for key, entry in aa_reg.items():
        for corr in entry["correlations"][:2]:
            feat = corr["feature"]
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: R² bar chart
    bars = ax1.barh(range(len(universes)), r2s, color=colors)
    ax1.set_yticks(range(len(universes)))
    ax1.set_yticklabels(universes, fontsize=9)
    ax1.set_xlabel("Adj. R² (6 LLM-discovered features)")
    ax1.set_title("Assistant axis is predictable\nfrom semantic features")
    ax1.set_xlim(0, 1)
    ax1.invert_yaxis()
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(
        handles=[Patch(facecolor="#377eb8", label="Franchise"), Patch(facecolor="#e41a1c", label="Mythology/Shakespeare")],
        loc="lower right", fontsize=8,
    )
    # Annotate values
    for i, v in enumerate(r2s):
        ax1.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    # Right: top feature frequency
    feat_names = [f[0] for f in top_features]
    feat_counts = [f[1] for f in top_features]
    ax2.barh(range(len(feat_names)), feat_counts, color="#4daf4a")
    ax2.set_yticks(range(len(feat_names)))
    ax2.set_yticklabels(feat_names, fontsize=8)
    ax2.set_xlabel("# universes where this is a top-2 feature")
    ax2.set_title("What predicts the assistant axis?\n(most common top features across universes)")
    ax2.invert_yaxis()

    fig.tight_layout()
    fig.savefig("results/intro_aa_features.png", dpi=150, bbox_inches="tight")
    print("Saved results/intro_aa_features.png")


def main():
    with open("results/fictional_character_analysis_filtered.pkl", "rb") as f:
        char_data = pickle.load(f)
    with open("data/role_vectors/qwen-3-32b_pca_layer32.pkl", "rb") as f:
        role_data = pickle.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = role_data["pca"]
    role_mean = role_pca.mean_
    chars_centered = activation_matrix - role_mean

    assistant_axis_all = torch.load("data/role_vectors/assistant_axis.pt", weights_only=True)
    assistant_axis = assistant_axis_all[32].float().numpy()
    aa_norm = assistant_axis / np.linalg.norm(assistant_axis)

    pca_fiction = make_eigenspectrum(chars_centered, role_pca)
    make_scatter(chars_centered, char_names, pca_fiction)
    make_heatmap_k1(chars_centered, char_names, role_pca, aa_norm)
    make_aa_figure(chars_centered, char_names, aa_norm)


if __name__ == "__main__":
    main()
