#!/usr/bin/env python3
"""
Show character responses to specific questions.

Usage:
    python scripts/show_responses.py harry_potter 0  # Show HP PC1 top questions
    python scripts/show_responses.py star_wars 0     # Show SW PC1 top questions
"""

import argparse
import json
import pickle
from pathlib import Path


def load_responses(responses_dir: Path, char_name: str, question_indices: list[int]):
    """Load specific question responses for a character."""
    jsonl_file = responses_dir / f"{char_name}.jsonl"
    if not jsonl_file.exists():
        return {}

    responses = {}
    with open(jsonl_file) as f:
        for line in f:
            data = json.loads(line)
            q_idx = data["question_index"]
            if q_idx in question_indices:
                # Get the assistant's response
                for msg in data["conversation"]:
                    if msg["role"] == "assistant":
                        responses[q_idx] = {
                            "question": data["question"],
                            "response": msg["content"],
                        }
                        break
    return responses


def main():
    parser = argparse.ArgumentParser(description="Show character responses")
    parser.add_argument("universe", type=str, help="Universe name (e.g., harry_potter)")
    parser.add_argument("pc_idx", type=int, default=0, help="PC index (0-based)")
    parser.add_argument(
        "--n-questions", type=int, default=3, help="Number of questions to show"
    )
    parser.add_argument(
        "--n-chars", type=int, default=2, help="Number of characters per extreme"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    responses_dir = repo_root / "outputs" / "qwen3-32b_20260211_002840" / "responses"

    # Load pre-computed data
    with open(repo_root / "results" / "pc_analysis.pkl", "rb") as f:
        pc_data = pickle.load(f)

    with open(repo_root / "results" / "question_correlations.pkl", "rb") as f:
        corr_data = pickle.load(f)

    if args.universe not in pc_data["universes"]:
        print(f"Universe {args.universe} not found")
        return

    u_pc = pc_data["universes"][args.universe]
    u_corr = corr_data[args.universe]

    # Get extreme characters
    import numpy as np

    scores = u_pc["pc_scores"][:, args.pc_idx]
    char_names_full = u_pc["char_names_full"]
    char_names_display = u_pc["characters"]
    sorted_idx = np.argsort(scores)

    high_chars = [
        (char_names_full[i], char_names_display[i], scores[i])
        for i in sorted_idx[-args.n_chars :][::-1]
    ]
    low_chars = [
        (char_names_full[i], char_names_display[i], scores[i])
        for i in sorted_idx[: args.n_chars]
    ]

    # Get top questions
    top_q = u_corr["top_questions"][args.pc_idx]
    pos_questions = [q[0] for q in top_q["positive"][: args.n_questions]]
    neg_questions = [q[0] for q in top_q["negative"][: args.n_questions]]
    all_questions = pos_questions + neg_questions

    var_pct = u_pc["variance_explained"][args.pc_idx] * 100
    print(
        f"=== {args.universe.replace('_', ' ').title()} PC{args.pc_idx + 1} ({var_pct:.1f}% variance) ===\n"
    )

    # Show high-correlating question responses
    print("=" * 80)
    print("TOP CORRELATING QUESTION")
    print("=" * 80)
    q_idx = pos_questions[0]
    q_text = top_q["positive"][0][2]
    print(f"\nQ{q_idx}: {q_text}\n")

    for char_full, char_display, score in high_chars:
        responses = load_responses(responses_dir, char_full, [q_idx])
        if q_idx in responses:
            print(f"--- {char_display} (PC score: {score:+.1f}) ---")
            print(responses[q_idx]["response"][:500])
            print("...\n")

    for char_full, char_display, score in low_chars:
        responses = load_responses(responses_dir, char_full, [q_idx])
        if q_idx in responses:
            print(f"--- {char_display} (PC score: {score:+.1f}) ---")
            print(responses[q_idx]["response"][:500])
            print("...\n")


if __name__ == "__main__":
    main()
