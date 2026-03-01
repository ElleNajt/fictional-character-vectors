#!/usr/bin/env python3
"""
Compute a vulnerability detection axis from CASTLE full activations.

vuln_axis = mean(vuln program response activations) - mean(clean program response activations)

Loads the per-program .pt files (seq_len, hidden_dim) and metadata.json,
averages response-phase tokens per program, then takes the difference.

Usage:
    python scripts/compute_vuln_axis.py
"""

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ACT_DIR = REPO_ROOT / "outputs" / "castle_full_activations"
META_PATH = ACT_DIR / "metadata.json"
OUTPUT_PATH = REPO_ROOT / "data" / "role_vectors" / "vuln_axis.pt"


def main():
    with open(META_PATH) as f:
        metadata = json.load(f)

    print(f"Loaded metadata for {len(metadata)} programs")

    vuln_means = []
    clean_means = []

    for i, entry in enumerate(metadata):
        act_path = ACT_DIR / entry["activation_file"]
        acts = torch.load(act_path, map_location="cpu", weights_only=True)  # (seq_len, hidden_dim)
        acts = acts.float()

        # Response tokens only
        prompt_len = entry["prompt_len"]
        response_acts = acts[prompt_len:]

        if response_acts.shape[0] == 0:
            print(f"  WARNING: {entry['test_id']} has no response tokens, skipping")
            continue

        program_mean = response_acts.mean(dim=0)

        if entry["vulnerable"]:
            vuln_means.append(program_mean)
        else:
            clean_means.append(program_mean)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(metadata)}")

    vuln_mean = torch.stack(vuln_means).mean(dim=0)
    clean_mean = torch.stack(clean_means).mean(dim=0)
    vuln_axis = vuln_mean - clean_mean

    print(f"\nVuln programs: {len(vuln_means)}")
    print(f"Clean programs: {len(clean_means)}")
    print(f"Vuln axis norm: {vuln_axis.norm():.4f}")
    print(f"Vuln mean norm: {vuln_mean.norm():.4f}")
    print(f"Clean mean norm: {clean_mean.norm():.4f}")

    # Also compute prompt-phase axis
    vuln_prompt_means = []
    clean_prompt_means = []

    for entry in metadata:
        act_path = ACT_DIR / entry["activation_file"]
        acts = torch.load(act_path, map_location="cpu", weights_only=True).float()
        prompt_acts = acts[:entry["prompt_len"]]
        program_mean = prompt_acts.mean(dim=0)

        if entry["vulnerable"]:
            vuln_prompt_means.append(program_mean)
        else:
            clean_prompt_means.append(program_mean)

    vuln_prompt_axis = torch.stack(vuln_prompt_means).mean(dim=0) - torch.stack(clean_prompt_means).mean(dim=0)
    print(f"\nPrompt-phase vuln axis norm: {vuln_prompt_axis.norm():.4f}")

    # Save both
    torch.save(vuln_axis, OUTPUT_PATH)
    print(f"\nSaved response-phase vuln axis to {OUTPUT_PATH}")

    prompt_output = OUTPUT_PATH.parent / "vuln_axis_prompt.pt"
    torch.save(vuln_prompt_axis, prompt_output)
    print(f"Saved prompt-phase vuln axis to {prompt_output}")


if __name__ == "__main__":
    main()
