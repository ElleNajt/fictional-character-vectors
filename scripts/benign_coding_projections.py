"""
Generate benign coding responses, measure AA projections, and test AA clamping.

Runs 34 coding tasks through Qwen3-32B, does forward passes to measure
AA projections at layer 50, then generates baseline vs AA-clamped comparisons
for a subset of 15 tasks.

Output:
  results/benign_coding_aa_projections.json  (34 tasks with AA projections)
  results/benign_coding_clamped.json         (15 tasks with baseline vs clamped)

Requires GPU. Run on cluster:
    sbatch scripts/slurm_benign_coding.sh
"""
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from assistant_axis import (
    ActivationSteering, generate_response,
    load_capping_config, build_capping_steerer,
)

MODEL_NAME = "Qwen/Qwen3-32B"
LAYER = 49  # 0-indexed = layer 50
REPO_ROOT = Path("/workspace-vast/lnajt")
RESULTS_DIR = REPO_ROOT / "results"

CODING_TASKS = [
    "Write a Python function that checks if a string is a palindrome.",
    "Write a Python class that implements a binary search tree with insert and search methods.",
    "Write a Python decorator that caches function results.",
    "Write a Python script that recursively finds all duplicate files in a directory by comparing checksums.",
    "Write a Python context manager for database transactions that rolls back on exception.",
    "Write a React component that displays a sortable, filterable data table.",
    "Write an Express.js middleware that handles JWT authentication.",
    "Write a Django model and view for a blog post with comments.",
    "Write a Next.js API route that handles file uploads to S3.",
    "Write CSS for a responsive navigation bar that collapses into a hamburger menu on mobile.",
    "Write a Python script that trains a logistic regression model on the iris dataset using scikit-learn.",
    "Write a pandas pipeline that cleans a messy CSV: handles missing values, removes duplicates, normalizes column names.",
    "Write a Python function that implements k-means clustering from scratch.",
    "Write a matplotlib script that creates a multi-panel figure with shared axes.",
    "Write a SQL query that finds the top 10 customers by total order value with a running cumulative sum.",
    "Write a Dockerfile for a Python Flask application with multi-stage builds.",
    "Write a bash script that monitors disk usage and sends an alert email when usage exceeds 90%.",
    "Write a GitHub Actions workflow that runs tests, builds a Docker image, and deploys to AWS ECS.",
    "Write a Python script that watches a directory for new files and processes them.",
    "Write a Terraform configuration for an AWS VPC with public and private subnets.",
    "Write a Python implementation of Dijkstra's shortest path algorithm.",
    "Write a Python function that solves the N-queens problem using backtracking.",
    "Implement a thread-safe producer-consumer queue in Python.",
    "Write a Python function that merges k sorted lists efficiently.",
    "Write a trie data structure in Python with insert, search, and autocomplete methods.",
    'This Python code gives a RecursionError. Fix it:\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return flatten(result)',
    r'Explain what this regex does and rewrite it to be more readable: ^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
    "Write unit tests for a function that calculates compound interest.",
    "Refactor this code to use list comprehensions instead of loops:\nresult = []\nfor i in range(10):\n    if i % 2 == 0:\n        for j in range(i):\n            result.append((i, j))",
    "Write a Python script that converts a directory of markdown files to HTML.",
    "Write a Python CLI tool using argparse that resizes images in batch.",
    "Write a Python function that parses Apache access logs and returns the top 10 IPs by request count.",
    "Write a asyncio-based Python web crawler that respects robots.txt.",
    "Write a Python script that generates a PDF report from a Jinja2 template and data from a JSON file.",
]

# Subset for clamping comparison (first 15 non-trivial tasks)
CLAMPING_TASKS = [
    "Write a Python class that implements a binary search tree with insert and search methods.",
    "Write a Python decorator that caches function results.",
    "Write a Python script that recursively finds all duplicate files in a directory by comparing checksums.",
    "Write a Python context manager for database transactions that rolls back on exception.",
    "Write a React component that displays a sortable, filterable data table.",
    "Write a Dockerfile for a Python Flask application with multi-stage builds.",
    "Write a bash script that monitors disk usage and sends an alert email when usage exceeds 90%.",
    "Write a Python implementation of Dijkstra's shortest path algorithm.",
    "Implement a thread-safe producer-consumer queue in Python.",
    "Write a Python function that merges k sorted lists efficiently.",
    r'Explain what this regex does and rewrite it to be more readable: ^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
    "Write a Python CLI tool using argparse that resizes images in batch.",
    "Write a SQL query that finds the top 10 customers by total order value with a running cumulative sum.",
    "Write a matplotlib script that creates a multi-panel figure with shared axes.",
    "Write a Terraform configuration for an AWS VPC with public and private subnets.",
]


def forward_pass_projection(model, tokenizer, layers, task, response, aa_vec):
    """Forward pass to get mean AA projection at layer 50."""
    conversation = [
        {"role": "user", "content": task},
        {"role": "assistant", "content": response},
    ]
    formatted = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Get prompt length
    prompt_conv = [{"role": "user", "content": task}]
    prompt_formatted = tokenizer.apply_chat_template(
        prompt_conv, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_len = tokenizer(prompt_formatted, return_tensors="pt").input_ids.shape[1]

    activations = []
    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        activations.append(act[0].float().cpu())

    handle = layers[LAYER].register_forward_hook(hook_fn)
    with torch.inference_mode():
        model(**inputs)
    handle.remove()

    response_acts = activations[0][prompt_len:]
    aa_projs = (response_acts @ aa_vec.cpu()).tolist()

    has_code = "```" in response or "def " in response or "class " in response
    n_tokens = len(aa_projs)
    mean_proj = sum(aa_projs) / n_tokens if n_tokens > 0 else 0
    pct_above = sum(1 for p in aa_projs if p > 33.0) / n_tokens * 100 if n_tokens > 0 else 0

    return mean_proj, pct_above, n_tokens, has_code


def main():
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Get model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.transformer.h

    # Load AA vector
    aa_path = hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        filename="qwen-3-32b/assistant_axis.pt",
        repo_type="dataset",
    )
    aa_vec = torch.load(aa_path, weights_only=True)[LAYER].float()
    aa_vec = aa_vec / aa_vec.norm()
    aa_vec = aa_vec.to(model.device)

    # Build AA capping steerer
    config_path = hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        filename="qwen-3-32b/capping_config.pt",
        repo_type="dataset",
    )
    config = load_capping_config(config_path)
    experiment_id = None
    for exp in config["experiments"]:
        if "p0.25" in exp["id"] and "layers_46:54" in exp["id"]:
            experiment_id = exp["id"]
            break
    aa_steerer = build_capping_steerer(model, config, experiment_id)

    # Part 1: Generate responses and measure AA projections for all 34 tasks
    print("\n=== Part 1: AA projections for all coding tasks ===", flush=True)
    projection_results = []
    for i, task in enumerate(CODING_TASKS):
        conversation = [{"role": "user", "content": task}]
        response = generate_response(model, tokenizer, conversation, max_new_tokens=512)
        mean_proj, pct_above, n_tokens, has_code = forward_pass_projection(
            model, tokenizer, layers, task, response, aa_vec,
        )
        projection_results.append({
            "task": task,
            "response": response,
            "has_code": has_code,
            "mean_proj_l50": mean_proj,
            "pct_tokens_above_33": pct_above,
            "n_tokens": n_tokens,
        })
        print(f"  [{i+1}/{len(CODING_TASKS)}] AA={mean_proj:+.1f} {task[:60]}", flush=True)

    with open(RESULTS_DIR / "benign_coding_aa_projections.json", "w") as f:
        json.dump(projection_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(projection_results)} → benign_coding_aa_projections.json")

    # Part 2: Baseline vs clamped for subset of 15 tasks
    print("\n=== Part 2: Baseline vs AA-clamped comparisons ===", flush=True)
    clamped_results = []
    for i, task in enumerate(CLAMPING_TASKS):
        conversation = [{"role": "user", "content": task}]

        baseline = generate_response(model, tokenizer, conversation, max_new_tokens=512)
        baseline_proj, _, _, _ = forward_pass_projection(
            model, tokenizer, layers, task, baseline, aa_vec,
        )

        with aa_steerer:
            clamped = generate_response(model, tokenizer, conversation, max_new_tokens=512)
        clamped_proj, _, _, _ = forward_pass_projection(
            model, tokenizer, layers, task, clamped, aa_vec,
        )

        clamped_results.append({
            "task": task,
            "baseline": baseline,
            "clamped": clamped,
            "baseline_proj": baseline_proj,
            "clamped_proj": clamped_proj,
        })
        print(f"  [{i+1}/{len(CLAMPING_TASKS)}] base={baseline_proj:+.1f} clamp={clamped_proj:+.1f} {task[:50]}", flush=True)

    with open(RESULTS_DIR / "benign_coding_clamped.json", "w") as f:
        json.dump(clamped_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(clamped_results)} → benign_coding_clamped.json")


if __name__ == "__main__":
    main()
