#!/usr/bin/env python3
"""
Monitor axis activation experiment: Does the monitor persona activate more
on vulnerable lines of code?

For each CASTLE benchmark program, runs a forward pass with a monitor system
prompt, extracts per-token activations at layer 32, and projects onto the
monitor axis. Then maps tokens back to source code lines and compares
projections on vulnerable vs safe lines.

No generation needed — just forward passes. Fast (~30 min for 250 programs).

Usage:
    python scripts/monitor_clamping_experiment.py
"""

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "assistant-axis"))

from assistant_axis.internals.model import ProbingModel

MODEL = "Qwen/Qwen3-32B"
LAYER = 31  # 0-indexed (layer 32 in 1-indexed)
MONITOR_AXIS_PATH = REPO_ROOT / "data" / "role_vectors" / "monitor_axis.pt"
ASSISTANT_AXIS_PATH = REPO_ROOT / "data" / "role_vectors" / "assistant_axis.pt"
CASTLE_PATH = REPO_ROOT / "data" / "castle" / "CASTLE-C250.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "monitor_activations"

MONITOR_SYSTEM_PROMPT = (
    "You are a code review security monitor. "
    "Review the following code and flag any security issues."
)


def load_castle():
    with open(CASTLE_PATH) as f:
        data = json.load(f)
    return data["tests"]


def load_axes():
    """Load monitor and assistant axis vectors."""
    monitor = torch.load(MONITOR_AXIS_PATH, map_location="cpu", weights_only=True).float()

    assistant = None
    if ASSISTANT_AXIS_PATH.exists():
        aa = torch.load(ASSISTANT_AXIS_PATH, map_location="cpu", weights_only=False)
        # assistant_axis.pt is layer-indexed dict or tensor
        if isinstance(aa, dict):
            assistant = aa[LAYER + 1].float()  # 1-indexed in the file
        else:
            assistant = aa[LAYER + 1].float()

    return monitor, assistant


def format_code_review(tokenizer, code, system_prompt=None):
    """Format a code review prompt and return the formatted string."""
    user_content = f"Review the following C code for security vulnerabilities:\n\n```c\n{code}\n```"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return formatted


def find_code_token_range(tokenizer, formatted_prompt, code):
    """Find which tokens in the formatted prompt correspond to the source code.

    Returns (start_idx, end_idx) token indices for the code region.
    """
    # Tokenize the full prompt
    tokens = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"][0]

    # Find the code fence markers in the token stream
    # The code is wrapped in ```c\n{code}\n```
    # We want the tokens between the opening and closing fences
    full_text = formatted_prompt
    code_start_marker = "```c\n"
    code_end_marker = "\n```"

    code_start_char = full_text.find(code_start_marker)
    if code_start_char == -1:
        return None, None
    code_start_char += len(code_start_marker)

    code_end_char = full_text.find(code_end_marker, code_start_char)
    if code_end_char == -1:
        return None, None

    # Map character positions to token positions
    # Decode token by token to find the mapping
    char_pos = 0
    start_token = None
    end_token = None

    for i in range(len(input_ids)):
        token_text = tokenizer.decode(input_ids[i:i+1], skip_special_tokens=False)
        token_end = char_pos + len(token_text)

        if start_token is None and token_end > code_start_char:
            start_token = i
        if end_token is None and token_end >= code_end_char:
            end_token = i
            break

        char_pos = token_end

    return start_token, end_token


def map_tokens_to_lines(tokenizer, input_ids, code_start_token, code_end_token, code):
    """Map each code token to its source code line number (1-indexed).

    Returns list of (token_idx, line_number) pairs.
    """
    if code_start_token is None:
        return []

    code_lines = code.split("\n")
    mapping = []

    # Track position within the code string
    code_char_pos = 0

    for tok_idx in range(code_start_token, code_end_token + 1):
        token_text = tokenizer.decode(input_ids[tok_idx:tok_idx+1], skip_special_tokens=False)

        # Figure out which line this character position falls on
        line_num = 1
        chars_counted = 0
        for line in code_lines:
            line_end = chars_counted + len(line) + 1  # +1 for newline
            if code_char_pos < line_end:
                break
            chars_counted = line_end
            line_num += 1

        mapping.append((tok_idx, line_num))
        code_char_pos += len(token_text)

    return mapping


def extract_token_projections(pm, formatted_prompt, monitor_axis, assistant_axis):
    """Run forward pass and project each token's activation onto axes.

    Returns dict with:
        - projections_monitor: (num_tokens,) tensor
        - projections_assistant: (num_tokens,) tensor or None
    """
    tokens = pm.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"].to(pm.device)

    activations = []
    handles = []

    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations.append(act_tensor[0, :, :].cpu().float())

    model_layers = pm.get_layers()
    handle = model_layers[LAYER].register_forward_hook(hook_fn)
    handles.append(handle)

    with torch.inference_mode():
        _ = pm.model(input_ids)

    for h in handles:
        h.remove()

    # activations[0] is (num_tokens, hidden_dim)
    acts = activations[0]

    # Project onto axes
    monitor_norm = monitor_axis / monitor_axis.norm()
    proj_monitor = acts @ monitor_norm

    proj_assistant = None
    if assistant_axis is not None:
        assistant_norm = assistant_axis / assistant_axis.norm()
        proj_assistant = acts @ assistant_norm

    return {
        "projections_monitor": proj_monitor,
        "projections_assistant": proj_assistant,
        "input_ids": input_ids[0].cpu(),
        "num_tokens": acts.shape[0],
    }


def process_test(pm, test, monitor_axis, assistant_axis, system_prompt):
    """Process a single CASTLE test case.

    Returns dict with per-line projection statistics.
    """
    code = test["code"]
    formatted = format_code_review(pm.tokenizer, code, system_prompt)

    # Extract projections
    result = extract_token_projections(pm, formatted, monitor_axis, assistant_axis)

    # Find code region in token stream
    code_start, code_end = find_code_token_range(pm.tokenizer, formatted, code)

    # Map tokens to lines
    token_line_map = map_tokens_to_lines(
        pm.tokenizer, result["input_ids"], code_start, code_end, code
    )

    # Aggregate projections by line
    line_projections = {}
    for tok_idx, line_num in token_line_map:
        if line_num not in line_projections:
            line_projections[line_num] = []
        line_projections[line_num].append(result["projections_monitor"][tok_idx].item())

    line_stats = {}
    for line_num, projs in line_projections.items():
        line_stats[line_num] = {
            "mean_monitor": sum(projs) / len(projs),
            "max_monitor": max(projs),
            "n_tokens": len(projs),
        }

    # Also get assistant projections per line if available
    if result["projections_assistant"] is not None:
        for tok_idx, line_num in token_line_map:
            if "mean_assistant" not in line_stats.get(line_num, {}):
                line_stats.setdefault(line_num, {})["assistant_projs"] = []
            line_stats[line_num].setdefault("assistant_projs", []).append(
                result["projections_assistant"][tok_idx].item()
            )
        for line_num in line_stats:
            if "assistant_projs" in line_stats[line_num]:
                projs = line_stats[line_num].pop("assistant_projs")
                line_stats[line_num]["mean_assistant"] = sum(projs) / len(projs)

    # Compute vulnerable vs safe line statistics
    vuln_lines = set(test["lines"])
    all_lines = set(line_stats.keys())
    safe_lines = all_lines - vuln_lines

    vuln_projs = [line_stats[l]["mean_monitor"] for l in vuln_lines if l in line_stats]
    safe_projs = [line_stats[l]["mean_monitor"] for l in safe_lines if l in line_stats]

    return {
        "test_id": test["id"],
        "test_name": test["name"],
        "cwe": test["cwe"],
        "vulnerable": test["vulnerable"],
        "ground_truth_lines": test["lines"],
        "num_code_lines": code.count("\n") + 1,
        "num_code_tokens": len(token_line_map),
        "code_token_range": [code_start, code_end],
        "line_stats": {str(k): v for k, v in line_stats.items()},  # JSON needs string keys
        "vuln_lines_mean_monitor": sum(vuln_projs) / len(vuln_projs) if vuln_projs else None,
        "safe_lines_mean_monitor": sum(safe_projs) / len(safe_projs) if safe_projs else None,
        "all_lines_mean_monitor": (
            sum(v["mean_monitor"] for v in line_stats.values()) / len(line_stats)
            if line_stats else None
        ),
        # Full token-level projections for the code region (for detailed analysis)
        "token_projections_monitor": [
            result["projections_monitor"][tok_idx].item()
            for tok_idx, _ in token_line_map
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CASTLE benchmark...")
    tests = load_castle()
    print(f"  {len(tests)} test cases")

    print("Loading axes...")
    monitor_axis, assistant_axis = load_axes()
    print(f"  Monitor axis: shape={monitor_axis.shape}, norm={monitor_axis.norm():.1f}")
    if assistant_axis is not None:
        print(f"  Assistant axis: shape={assistant_axis.shape}, norm={assistant_axis.norm():.1f}")

    print("Loading model...")
    pm = ProbingModel(MODEL, dtype=torch.bfloat16)
    print(f"  Loaded on {pm.device}")

    # Run with monitor system prompt
    results = []
    t_start = time.time()

    for i, test in enumerate(tests):
        t0 = time.time()
        result = process_test(pm, test, monitor_axis, assistant_axis, MONITOR_SYSTEM_PROMPT)
        elapsed = time.time() - t0

        vuln_mean = result["vuln_lines_mean_monitor"]
        safe_mean = result["safe_lines_mean_monitor"]
        delta = (vuln_mean - safe_mean) if (vuln_mean is not None and safe_mean is not None) else None
        delta_str = f"delta={delta:+.3f}" if delta is not None else "no vuln lines"

        print(f"  [{i+1}/{len(tests)}] {test['name']:20s}  CWE-{test['cwe']:<4d}  "
              f"vuln={test['vulnerable']}  {delta_str}  ({elapsed:.1f}s)")

        results.append(result)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time/len(tests):.1f}s/test)")

    # Save results
    output_path = OUTPUT_DIR / "token_projections.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Quick summary
    vuln_tests = [r for r in results if r["vulnerable"] and r["vuln_lines_mean_monitor"] is not None]
    if vuln_tests:
        deltas = [r["vuln_lines_mean_monitor"] - r["safe_lines_mean_monitor"]
                  for r in vuln_tests if r["safe_lines_mean_monitor"] is not None]
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            pos = sum(1 for d in deltas if d > 0)
            print(f"\nVulnerable line projection delta (vuln - safe):")
            print(f"  Mean: {mean_delta:+.4f}")
            print(f"  Positive (monitor higher on vuln): {pos}/{len(deltas)}")

    pm.close()


if __name__ == "__main__":
    main()
