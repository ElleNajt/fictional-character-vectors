#!/usr/bin/env python3
"""
Derive adversarial_layer50_projections.json from adversarial_token_projections.json.

Extracts mean AA projection per (character, condition, question) from the
per-token projection data.

Input:  results/adversarial_token_projections.json
Output: results/adversarial_layer50_projections.json

Run from repo root:
    python blogpost/scripts/derive_adversarial_projections.py
"""
import json
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    input_path = repo_root / "results" / "adversarial_token_projections.json"
    output_path = repo_root / "results" / "adversarial_layer50_projections.json"

    with open(input_path) as f:
        token_data = json.load(f)

    results = []
    for entry in token_data:
        results.append({
            "character": entry["character"],
            "condition": entry["condition"],
            "question": entry["question"],
            "mean_proj_l50": entry["mean_proj"],
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Derived {len(results)} entries → {output_path}")


if __name__ == "__main__":
    main()
