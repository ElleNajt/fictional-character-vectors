#!/usr/bin/env python3
"""
Compute a vulnerability-line axis from CASTLE full activations.

For each vulnerable program, maps ground_truth_lines to token positions,
then computes:
  vuln_line_axis = mean(activations at vulnerable lines) - mean(activations at non-vulnerable lines)

Only uses prompt-phase tokens within the code block.
"""

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ACT_DIR = REPO_ROOT / "outputs" / "castle_full_activations"
META_PATH = ACT_DIR / "metadata.json"
OUTPUT_PATH = REPO_ROOT / "data" / "role_vectors" / "vuln_line_axis.pt"


def find_code_block_range(tokens):
    """Find token indices for the code block (between ```c and ```)."""
    start = None
    for i, t in enumerate(tokens):
        if t == '```' and i + 1 < len(tokens) and tokens[i + 1] in ('c', 'c\n'):
            # Code block starts after ```c\n
            start = i + 2
            # If tokens[i+1] == 'c', the newline might be the next token
            if tokens[i + 1] == 'c' and i + 2 < len(tokens):
                if tokens[i + 2] == '\n':
                    start = i + 3
                else:
                    start = i + 2
            break

    if start is None:
        return None, None

    # Find closing ```
    end = None
    for i in range(start, len(tokens)):
        if tokens[i] == '```' or tokens[i].startswith('```'):
            end = i
            break

    return start, end


def map_lines_to_tokens(tokens, code_start, code_end, target_lines):
    """Map source line numbers to token indices within the code block.

    Returns (vuln_token_indices, non_vuln_token_indices) within [code_start, code_end).
    """
    target_set = set(target_lines)
    current_line = 1  # Source lines are 1-indexed
    vuln_tokens = []
    non_vuln_tokens = []

    for i in range(code_start, code_end):
        tok = tokens[i]

        if current_line in target_set:
            vuln_tokens.append(i)
        else:
            non_vuln_tokens.append(i)

        # Count newlines in this token to track line number
        newline_count = tok.count('\n')
        current_line += newline_count

    return vuln_tokens, non_vuln_tokens


def main():
    with open(META_PATH) as f:
        metadata = json.load(f)

    vuln_entries = [e for e in metadata if e['vulnerable']]
    print(f"Vulnerable programs: {len(vuln_entries)}")

    all_vuln_acts = []
    all_nonvuln_acts = []
    skipped = 0

    for i, entry in enumerate(vuln_entries):
        tokens = entry['tokens']
        code_start, code_end = find_code_block_range(tokens)

        if code_start is None:
            print(f"  WARNING: No code block found in {entry['test_id']}, skipping")
            skipped += 1
            continue

        vuln_tok_idx, nonvuln_tok_idx = map_lines_to_tokens(
            tokens, code_start, code_end, entry['ground_truth_lines']
        )

        if not vuln_tok_idx or not nonvuln_tok_idx:
            print(f"  WARNING: {entry['test_id']} has {len(vuln_tok_idx)} vuln tokens, {len(nonvuln_tok_idx)} non-vuln tokens, skipping")
            skipped += 1
            continue

        act_path = ACT_DIR / entry["activation_file"]
        acts = torch.load(act_path, map_location="cpu", weights_only=True).float()

        vuln_act = acts[vuln_tok_idx].mean(dim=0)
        nonvuln_act = acts[nonvuln_tok_idx].mean(dim=0)

        all_vuln_acts.append(vuln_act)
        all_nonvuln_acts.append(nonvuln_act)

        if (i + 1) % 30 == 0:
            print(f"  Processed {i+1}/{len(vuln_entries)}, "
                  f"avg vuln tokens: {sum(len(e) for e in [vuln_tok_idx]):.0f}, "
                  f"avg non-vuln tokens: {sum(len(e) for e in [nonvuln_tok_idx]):.0f}")

    print(f"\nUsed {len(all_vuln_acts)} programs (skipped {skipped})")

    vuln_mean = torch.stack(all_vuln_acts).mean(dim=0)
    nonvuln_mean = torch.stack(all_nonvuln_acts).mean(dim=0)
    vuln_line_axis = vuln_mean - nonvuln_mean

    print(f"Vuln line mean norm: {vuln_mean.norm():.4f}")
    print(f"Non-vuln line mean norm: {nonvuln_mean.norm():.4f}")
    print(f"Vuln line axis norm: {vuln_line_axis.norm():.4f}")

    torch.save(vuln_line_axis, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
