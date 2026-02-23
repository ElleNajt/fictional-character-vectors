"""
Question subset sweep: how few questions reproduce the full-battery PCA?

Uses pre-computed per-question projections (question_projections.pkl).
For each subset size, draws 100 random subsets of questions, computes
PCA on the subset means, and correlates PC1 scores with full-battery PC1.

Output: results/question_subset_sweep.json
"""

import json
import pickle

import numpy as np


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

        # Full-battery scores: mean across all 240 questions, then PC1
        full_mean = projs.mean(axis=1)  # (n_chars, n_pcs)
        # PC1 of full battery = first column (already PCA-projected)
        full_pc1 = full_mean[:, 0]

        sweep = []
        for size in subset_sizes:
            if size > n_questions:
                continue
            correlations = []
            for _ in range(n_draws):
                q_idx = rng.choice(n_questions, size=size, replace=False)
                subset_mean = projs[:, q_idx, :].mean(axis=1)  # (n_chars, n_pcs)
                # Correlate subset PC1 with full PC1
                # (PCA sign is arbitrary, use absolute correlation)
                r = abs(np.corrcoef(subset_mean[:, 0], full_pc1)[0, 1])
                correlations.append(float(r))

            correlations = sorted(correlations)
            sweep.append(
                {
                    "size": size,
                    "median": float(np.median(correlations)),
                    "mean": float(np.mean(correlations)),
                    "p5": float(np.percentile(correlations, 5)),
                    "p25": float(np.percentile(correlations, 25)),
                    "p75": float(np.percentile(correlations, 75)),
                    "p95": float(np.percentile(correlations, 95)),
                    "min": float(min(correlations)),
                }
            )

        results[universe] = {
            "n_chars": n_chars,
            "sweep": sweep,
        }

        # Summary line
        for s in sweep:
            if s["size"] in [1, 5, 10, 20, 50, 120]:
                print(
                    f"  {universe:20s}  n={s['size']:3d}  "
                    f"median={s['median']:.4f}  "
                    f"p5={s['p5']:.4f}  p95={s['p95']:.4f}"
                )

    with open("results/question_subset_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/question_subset_sweep.json")


if __name__ == "__main__":
    main()
