"""
Principal angle heatmaps: cross-universe subspace alignment.

For each pair of universes (plus role PCs and assistant axis), compute
the principal angles between their top-k PC subspaces (k=1,2,3).

The k-th principal angle measures how well the k-th best-aligned direction
in subspace A matches subspace B, after accounting for the first k-1.
Cosine of the k-th principal angle = k-th singular value of Q_A^T Q_B
where Q_A, Q_B are orthonormal bases.

For k=1, this is just |cosine(PC1_A, PC1_B)|.

Output: results/principal_angle_heatmaps.png
"""

import pickle
import sys

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


def get_universe_indices(char_names, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def principal_angles_cosines(A, B):
    """Cosines of principal angles between subspaces spanned by rows of A and B.

    A: (k, d) orthonormal rows
    B: (m, d) orthonormal rows
    Returns: array of min(k, m) cosines, sorted descending.
    """
    M = A @ B.T
    s = svd(M, compute_uv=False)
    return np.clip(s, 0, 1)


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

    assistant_axis_all = torch.load(
        "data/role_vectors/assistant_axis.pt", weights_only=True
    )
    assistant_axis = assistant_axis_all[32].float().numpy()
    aa_norm = assistant_axis / np.linalg.norm(assistant_axis)

    # Fit within-universe PCAs
    universe_pcas = {}
    for name, prefixes in ALL_UNIVERSES.items():
        idx = get_universe_indices(char_names, prefixes)
        n_comp = min(len(idx) - 1, 10)
        pca = SkPCA(n_components=n_comp).fit(chars_centered[idx])
        universe_pcas[name] = pca.components_  # (n_comp, d)

    # Labels: universes + role PCs + assistant axis
    labels = list(ALL_UNIVERSES.keys()) + ["Role PCs", "Asst. Axis"]
    n = len(labels)

    # For each k=1,2,3: compute principal angle cosines
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    for ki, k in enumerate([1, 2, 3]):
        mat = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(n):
                # Get subspace basis (top k components)
                if labels[i] == "Asst. Axis":
                    if k > 1:
                        continue
                    basis_i = aa_norm.reshape(1, -1)
                elif labels[i] == "Role PCs":
                    basis_i = role_pca.components_[:k]
                else:
                    comps = universe_pcas[labels[i]]
                    if k > comps.shape[0]:
                        continue
                    basis_i = comps[:k]

                if labels[j] == "Asst. Axis":
                    if k > 1:
                        continue
                    basis_j = aa_norm.reshape(1, -1)
                elif labels[j] == "Role PCs":
                    basis_j = role_pca.components_[:k]
                else:
                    comps = universe_pcas[labels[j]]
                    if k > comps.shape[0]:
                        continue
                    basis_j = comps[:k]

                cosines = principal_angles_cosines(basis_i, basis_j)
                # The k-th principal angle cosine (0-indexed: k-1)
                # For k=1 subspaces, there's only 1 angle
                # For k=2, show the 2nd (worst) principal angle
                # For k=3, show the 3rd (worst) principal angle
                # This shows how well the FULL k-dim subspace aligns
                mat[i, j] = cosines[min(k, len(cosines)) - 1]

        ax = axes[ki]
        # Mask NaN for display
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]) and i != j:
                    color = "white" if mat[i, j] < 0.4 else "black"
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=color,
                    )

        if k == 1:
            ax.set_title(
                f"1st principal angle cosine\n(= |cos(PC1, PC1)|)", fontsize=10
            )
        else:
            ax.set_title(
                f"{k}th principal angle cosine\n(worst alignment in {k}D subspace)",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig("results/principal_angle_heatmaps.png", dpi=150, bbox_inches="tight")
    print("Saved results/principal_angle_heatmaps.png")


if __name__ == "__main__":
    main()
