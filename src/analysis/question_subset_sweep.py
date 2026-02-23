"""
Question subset sweep: how few questions reproduce the full-battery PCA?

Uses pre-computed per-question projections (question_projections.pkl).
For each subset size, draws 200 random subsets of questions, computes
mean projections, and correlates with the full-battery scores for each PC.

Output: results/question_subset_sweep.json
        results/question_subset_sweep.png
"""

import json
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open("results/question_projections.pkl", "rb") as f:
        projections_data = pickle.load(f)

    subset_sizes = [1, 2, 3, 5, 10, 15, 20, 30, 50, 80, 100, 120, 160, 200, 240]
    n_draws = 200
    rng = np.random.default_rng(42)

    results = {}

    for universe, udata in projections_data.items():
        projs = np.array(udata["projections"])  # (n_chars, 240, n_pcs)
        n_chars, n_questions, n_pcs = projs.shape
        full_mean = projs.mean(axis=1)  # (n_chars, n_pcs)

        pc_sweeps = {}
        for pc_i in range(n_pcs):
            full_scores = full_mean[:, pc_i]
            sweep = []
            for size in subset_sizes:
                if size > n_questions:
                    continue
                correlations = []
                for _ in range(n_draws):
                    q_idx = rng.choice(n_questions, size=size, replace=False)
                    subset_mean = projs[:, q_idx, :].mean(axis=1)
                    r = abs(np.corrcoef(subset_mean[:, pc_i], full_scores)[0, 1])
                    correlations.append(float(r))

                sweep.append(
                    {
                        "size": size,
                        "median": float(np.median(correlations)),
                        "p5": float(np.percentile(correlations, 5)),
                        "p25": float(np.percentile(correlations, 25)),
                        "p75": float(np.percentile(correlations, 75)),
                        "p95": float(np.percentile(correlations, 95)),
                    }
                )
            pc_sweeps[f"PC{pc_i + 1}"] = sweep

        results[universe] = {"n_chars": n_chars, "n_pcs": n_pcs, "sweeps": pc_sweeps}

        # Summary
        for pc_i in range(n_pcs):
            for s in pc_sweeps[f"PC{pc_i + 1}"]:
                if s["size"] == 5:
                    print(
                        f"  {universe:20s}  PC{pc_i + 1}  n=5  "
                        f"median={s['median']:.4f}  p5={s['p5']:.4f}"
                    )

    with open("results/question_subset_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/question_subset_sweep.json")

    # --- Plot: median correlation vs subset size, one line per PC ---
    # Aggregate across universes (median of medians)
    n_pcs_all = min(d["n_pcs"] for d in results.values())
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_pcs_all))

    for pc_i in range(n_pcs_all):
        pc_label = f"PC{pc_i + 1}"
        # For each size, collect median across universes
        size_to_medians = {}
        size_to_p5s = {}
        for universe, udata in results.items():
            for s in udata["sweeps"][pc_label]:
                size_to_medians.setdefault(s["size"], []).append(s["median"])
                size_to_p5s.setdefault(s["size"], []).append(s["p5"])

        sizes = sorted(size_to_medians.keys())
        medians = [np.median(size_to_medians[s]) for s in sizes]
        p5s = [np.median(size_to_p5s[s]) for s in sizes]

        ax.plot(sizes, medians, "o-", color=colors[pc_i], label=pc_label, markersize=3)
        ax.fill_between(sizes, p5s, medians, color=colors[pc_i], alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("# questions in random subset")
    ax.set_ylabel("correlation with full-battery ranking")
    ax.set_ylim(0.5, 1.01)
    ax.axhline(0.99, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(fontsize=9)
    ax.set_title("Question subset recovery (median across 12 universes, 200 draws)")

    plt.tight_layout()
    plt.savefig("results/question_subset_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/question_subset_sweep.png")


if __name__ == "__main__":
    main()
