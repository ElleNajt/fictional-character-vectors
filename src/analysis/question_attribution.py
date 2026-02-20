"""Question attribution along residual PC1 for each universe.

Decomposes each character's PC score into per-question contributions
using the linear identity: score = (1/240) * sum_q (residual_q . pc_direction).

Outputs:
  results/question_attribution_cumvar.png  - cumulative variance plots
  results/question_attribution.txt         - summary table + per-character attributions
"""

import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA as SkPCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- Setup ---
with open("results/fictional_character_analysis_filtered.pkl", "rb") as f:
    char_data = pickle.load(f)

questions = []
with open("assistant-axis/data/extraction_questions.jsonl") as f:
    for line in f:
        questions.append(json.loads(line)["question"])

char_names = char_data["character_names"]
activation_matrix = char_data["activation_matrix"]

with open("data/role_vectors/qwen-3-32b_pca_layer32.pkl", "rb") as f:
    lu_data = pickle.load(f)
role_pca = lu_data["pca"]
scaler = lu_data["scaler"]

chars_scaled = scaler.transform(activation_matrix)
chars_in_role_space = role_pca.transform(chars_scaled)
reconstructed = chars_in_role_space @ role_pca.components_
residuals = chars_scaled - reconstructed

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

activations_dir = Path("outputs/qwen3-32b_20260211_002840/activations")


def get_universe_indices(prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def load_per_question_activations(char_name, layer=32):
    pt_file = activations_dir / f"{char_name}.pt"
    if not pt_file.exists():
        return None
    data = torch.load(pt_file, map_location="cpu")
    question_activations = []
    for q_idx in range(240):
        q_acts = []
        for p_idx in range(5):
            key = f"pos_p{p_idx}_q{q_idx}"
            if key in data:
                act = data[key]
                if isinstance(act, torch.Tensor):
                    act = act.float().numpy()
                if len(act.shape) == 2:
                    q_acts.append(act[layer - 1])
                else:
                    q_acts.append(act)
        if q_acts:
            question_activations.append(np.mean(q_acts, axis=0))
    return np.array(question_activations) if question_activations else None


# --- Main computation ---
print("Computing question attributions...")

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

universe_results = {}

for ax_idx, (universe, prefixes) in enumerate(ALL_UNIVERSES.items()):
    indices = get_universe_indices(prefixes)
    if len(indices) < 20:
        continue

    print(f"  {universe} ({len(indices)} characters)...")
    u_residuals = residuals[indices]
    u_names = [char_names[i] for i in indices]

    u_pca = SkPCA(n_components=1)
    scores = u_pca.fit_transform(u_residuals)[:, 0]
    pc1_dir = u_pca.components_[0]

    all_projections = []
    valid_names = []
    for cname in u_names:
        acts = load_per_question_activations(cname)
        if acts is not None and len(acts) == 240:
            acts_scaled = scaler.transform(acts)
            in_role = role_pca.transform(acts_scaled)
            acts_resid = acts_scaled - in_role @ role_pca.components_
            projections = acts_resid @ pc1_dir
            all_projections.append(projections)
            valid_names.append(cname)

    all_projections = np.array(all_projections)
    question_variance = np.var(all_projections, axis=0)

    sorted_var = np.sort(question_variance)[::-1]
    cumvar = np.cumsum(sorted_var) / sorted_var.sum()

    n50 = int(np.searchsorted(cumvar, 0.5) + 1)
    n80 = int(np.searchsorted(cumvar, 0.8) + 1)
    n90 = int(np.searchsorted(cumvar, 0.9) + 1)

    # Plot
    ax = axes[ax_idx]
    ax.plot(range(1, 241), cumvar, "b-", linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"{universe}\n50%@{n50}, 80%@{n80}, 90%@{n90} questions")
    ax.set_xlabel("# questions (sorted)")
    ax.set_ylabel("cumulative variance fraction")
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 1.05)

    top_q_idx = np.argsort(question_variance)[::-1][:10]
    sorted_idx = np.argsort(scores)
    universe_results[universe] = {
        "question_variance": question_variance,
        "all_projections": all_projections,
        "valid_names": valid_names,
        "scores": scores,
        "top_q_idx": top_q_idx,
        "n50": n50,
        "n80": n80,
        "n90": n90,
        "high": [
            u_names[i].split("__")[-1].replace("_", " ").title()
            for i in sorted_idx[-5:]
        ],
        "low": [
            u_names[i].split("__")[-1].replace("_", " ").title() for i in sorted_idx[:5]
        ],
    }

for i in range(len(universe_results), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig("results/question_attribution_cumvar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/question_attribution_cumvar.png")

# --- Text output ---
out = []

# Summary table
rows = []
for universe, data in universe_results.items():
    qv = data["question_variance"]
    top1_pct = qv[data["top_q_idx"][0]] / qv.sum() * 100
    rows.append(
        {
            "Universe": universe,
            "n50": data["n50"],
            "n80": data["n80"],
            "n90": data["n90"],
            "Top Q %": f"{top1_pct:.1f}%",
        }
    )

df = pd.DataFrame(rows)
table = df.to_string(index=False)
out.append("=== Variance Concentration ===\n")
out.append(table)
print(table)

# --- Extreme character analysis ---
# For the top 3 and bottom 3 characters in each universe:
# - How much of their PC score comes from the top k questions?
# - Are the top questions worth reading, or is the signal too diffuse?
out.append("\n\n=== Extreme Character Attribution Analysis ===\n")
out.append("For each extreme character: cumulative fraction of total PC score")
out.append("from top-k questions (sorted by |projection|).\n")

print("\n=== Extreme Character Attribution Analysis ===")

for universe, data in universe_results.items():
    projs = data["all_projections"]
    names = data["valid_names"]

    sorted_chars = sorted(range(len(names)), key=lambda i: projs[i].mean())

    header = f"\n--- {universe} ---"
    out.append(header)
    print(header)

    # Top 3 and bottom 3
    extreme_indices = sorted_chars[-3:][::-1] + sorted_chars[:3]

    for char_idx in extreme_indices:
        cname = names[char_idx].split("__")[-1].replace("_", " ").title()
        char_projs = projs[char_idx]
        total_score = char_projs.sum()  # = 240 * mean_proj

        # Sort by |projection| descending
        abs_order = np.argsort(np.abs(char_projs))[::-1]
        cumsum = np.cumsum(char_projs[abs_order])

        # Fraction of total score from top k
        top5_frac = cumsum[4] / total_score if total_score != 0 else 0
        top10_frac = cumsum[9] / total_score if total_score != 0 else 0
        top20_frac = cumsum[19] / total_score if total_score != 0 else 0

        # Effective N (on absolute values)
        abs_projs = np.abs(char_projs)
        p = abs_projs / abs_projs.sum()
        eff_n = 1.0 / (p**2).sum()

        line = (
            f"  {cname:30s}  proj={char_projs.mean():+6.1f}  "
            f"eff_N={eff_n:5.0f}  "
            f"top5={top5_frac:4.0%}  top10={top10_frac:4.0%}  top20={top20_frac:4.0%}"
        )
        out.append(line)
        print(line)

        # Show the top 5 questions for this character
        for rank, qi in enumerate(abs_order[:5]):
            q_frac = char_projs[qi] / total_score * 100 if total_score != 0 else 0
            qline = (
                f"      {char_projs[qi]:+6.1f} ({q_frac:+4.1f}%)  {questions[qi][:65]}"
            )
            out.append(qline)
            print(qline)

Path("results/question_attribution.txt").write_text("\n".join(out))
print("\nSaved results/question_attribution.txt")
