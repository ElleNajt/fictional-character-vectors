"""Generate HumanEval completions with and without AA clamping.

Generates baseline and AA-clamped completions for all 164 HumanEval problems.
Saves in HumanEval JSONL format for local evaluation.
"""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from assistant_axis import load_capping_config, build_capping_steerer, generate_response
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen3-32B"
REPO_ROOT = Path('/workspace-vast/lnajt')
RESULTS_DIR = REPO_ROOT / 'results'

print("Loading HumanEval problems...", flush=True)
ds = load_dataset("openai/openai_humaneval", split="test")
print(f"Loaded {len(ds)} problems", flush=True)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Building AA capping steerer...", flush=True)
capping_config_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="qwen-3-32b/capping_config.pt", repo_type="dataset")
capping_config = load_capping_config(capping_config_path)
aa_steerer = build_capping_steerer(model, capping_config, 'layers_46:54-p0.25')

baseline_results = []
clamped_results = []

for i, problem in enumerate(ds):
    task_id = problem["task_id"]
    prompt = problem["prompt"]

    conversation = [
        {"role": "user", "content": f"Complete the following Python function. Return ONLY the function body, no explanation.\n\n{prompt}"}
    ]

    print(f"\n[{i+1}/{len(ds)}] {task_id}", flush=True)

    # Baseline
    baseline = generate_response(model, tokenizer, conversation, max_new_tokens=512)
    baseline_results.append({"task_id": task_id, "completion": baseline})
    print(f"  baseline: {len(baseline)} chars", flush=True)

    # AA clamped
    with aa_steerer:
        clamped = generate_response(model, tokenizer, conversation, max_new_tokens=512)
    clamped_results.append({"task_id": task_id, "completion": clamped})
    print(f"  clamped:  {len(clamped)} chars", flush=True)

    # Save incrementally
    if (i + 1) % 10 == 0 or i == len(ds) - 1:
        for name, results in [("baseline", baseline_results), ("aa_clamped", clamped_results)]:
            path = RESULTS_DIR / f'humaneval_{name}.jsonl'
            with open(path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"  Saved {i+1} results", flush=True)

print(f"\nDone. Saved {len(baseline_results)} baseline and {len(clamped_results)} clamped completions.", flush=True)
