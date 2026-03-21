#!/usr/bin/env python3
"""
Compute question-PC correlations using pre-computed projections.

Uses question_projections.pkl and pc_analysis.pkl to find which questions
correlate most strongly with each PC.

Output: results/question_correlations.pkl
  {
    'harry_potter': {
      'correlations': np.array (n_questions=240, n_pcs),  # correlation of each question with each PC
      'top_questions': {
          0: [(q_idx, corr, question_text), ...],  # top 10 for PC1
          1: [...],  # top 10 for PC2
          ...
      }
    },
    ...
  }
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description="Compute question-PC correlations")
    parser.add_argument(
        "--output", type=str, default="results/question_correlations.pkl"
    )
    parser.add_argument(
        "--n-top", type=int, default=10, help="Number of top questions per PC"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    # Load pre-computed data
    print("Loading pre-computed data...")
    with open(repo_root / "results" / "question_projections.pkl", "rb") as f:
        proj_data = pickle.load(f)

    with open(repo_root / "results" / "pc_analysis.pkl", "rb") as f:
        pc_data = pickle.load(f)

    # Load questions (JSONL format)
    questions_file = (
        repo_root / "assistant-axis" / "data" / "extraction_questions.jsonl"
    )
    questions = []
    with open(questions_file) as f:
        for line in f:
            q = json.loads(line)
            questions.append(q["question"])

    output_data = {}

    for universe_key, u_proj in proj_data.items():
        print(f"\n=== {universe_key} ===")

        char_names = u_proj["char_names"]
        projections = u_proj["projections"]  # (n_chars, 240, n_pcs)
        n_chars, n_questions, n_pcs = projections.shape

        # Get PC scores from pc_analysis
        if universe_key not in pc_data["universes"]:
            print(f"  No PC data for {universe_key}, skipping")
            continue

        pc_universe = pc_data["universes"][universe_key]
        pc_scores = pc_universe["pc_scores"]  # (n_chars, n_pcs) array
        pc_char_names = pc_universe[
            "char_names_full"
        ]  # internal names matching projections

        # Both should have the same characters in the same order now
        # But let's verify and align if needed
        proj_name_set = set(char_names)
        pc_name_set = set(pc_char_names)

        # Find common characters and their indices
        common_names = proj_name_set & pc_name_set
        if len(common_names) < 10:
            print(f"  Only {len(common_names)} common chars, skipping")
            continue

        # Build index mappings
        proj_indices = []
        pc_indices = []
        for name in common_names:
            proj_indices.append(char_names.index(name))
            pc_indices.append(pc_char_names.index(name))

        projections = projections[proj_indices]  # (n_common, 240, n_pcs)
        pc_scores = pc_scores[pc_indices]  # (n_common, n_pcs)
        n_chars = len(common_names)

        print(f"  {n_chars} characters matched")

        # Compute correlations: for each question, correlate its projection with PC score
        # projections[:, q, pc] is the projection of question q onto PC pc for each char
        # pc_scores[:, pc] is the PC score for each char
        #
        # For each PC, we want: corr(projections[:, q, pc], pc_scores[:, pc])
        # This tells us which questions "agree" with where a character falls on the PC

        correlations = np.zeros((n_questions, n_pcs))

        for pc_idx in range(n_pcs):
            pc_score_vec = pc_scores[:, pc_idx]  # (n_chars,)
            for q_idx in range(n_questions):
                q_proj_vec = projections[:, q_idx, pc_idx]  # (n_chars,)
                r, _ = stats.pearsonr(q_proj_vec, pc_score_vec)
                correlations[q_idx, pc_idx] = r

        # Find top questions for each PC
        top_questions = {}
        for pc_idx in range(n_pcs):
            corrs = correlations[:, pc_idx]

            # Get top positive and negative correlations
            sorted_indices = np.argsort(corrs)
            top_pos = sorted_indices[-args.n_top :][::-1]  # highest first
            top_neg = sorted_indices[: args.n_top]  # lowest first

            top_questions[pc_idx] = {
                "positive": [
                    (int(q_idx), float(corrs[q_idx]), questions[q_idx])
                    for q_idx in top_pos
                ],
                "negative": [
                    (int(q_idx), float(corrs[q_idx]), questions[q_idx])
                    for q_idx in top_neg
                ],
            }

            print(
                f"  PC{pc_idx + 1} top positive: r={corrs[top_pos[0]]:.3f} - {questions[top_pos[0]][:50]}..."
            )
            print(
                f"  PC{pc_idx + 1} top negative: r={corrs[top_neg[0]]:.3f} - {questions[top_neg[0]][:50]}..."
            )

        output_data[universe_key] = {
            "correlations": correlations,
            "top_questions": top_questions,
            "n_chars": n_chars,
            "variance_explained": u_proj["variance_explained"],
        }

    # Save
    output_path = repo_root / args.output
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
