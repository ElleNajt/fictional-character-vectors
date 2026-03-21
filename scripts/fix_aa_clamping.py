"""
Re-generate aa_clamped responses with the correct capping experiment.

The original adversarial_clamping.py selected layers_32:36-p0.25 (first match)
instead of layers_46:54-p0.25. This script fixes all entries by re-running
only the aa_clamped generation with the correct config.

Usage:
    python scripts/fix_aa_clamping.py [--resume]
"""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistant_axis import ActivationSteering, generate_response
from huggingface_hub import hf_hub_download
from assistant_axis import load_capping_config, build_capping_steerer

MODEL_NAME = "Qwen/Qwen3-32B"
REPO_ROOT = Path('/workspace-vast/lnajt')
RESULTS_PATH = REPO_ROOT / 'results' / 'adversarial_clamping_comparison.json'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

with open(RESULTS_PATH) as f:
    results = json.load(f)

print(f"Loaded {len(results)} entries")

# Build AA steerer with CORRECT experiment
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

capping_config_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="qwen-3-32b/capping_config.pt", repo_type="dataset")
capping_config = load_capping_config(capping_config_path)

experiment_id = 'layers_46:54-p0.25'
print(f"AA experiment: {experiment_id}")
aa_steerer = build_capping_steerer(model, capping_config, experiment_id)

# Re-generate aa_clamped for all entries
for i, entry in enumerate(results):
    # Skip if already fixed (marked by presence of _aa_fix flag)
    if args.resume and entry.get('_aa_fix'):
        continue

    conversation = [
        {"role": "system", "content": entry["system_prompt"]},
        {"role": "user", "content": entry["question"]},
    ]

    with aa_steerer:
        aa_clamped = generate_response(model, tokenizer, conversation, max_new_tokens=512)

    entry['aa_clamped'] = aa_clamped
    entry['_aa_fix'] = True

    print(f"  [{i+1}/{len(results)}] {entry['character']:30s} {entry['question'][:50]}")
    print(f"    {aa_clamped[:120]}")

    if (i + 1) % 10 == 0:
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  (saved)")

# Remove fix markers and save
for entry in results:
    entry.pop('_aa_fix', None)

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(results)} fixed entries to {RESULTS_PATH}")
