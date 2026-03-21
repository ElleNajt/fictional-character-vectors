#!/bin/bash
#SBATCH --job-name=forseti_clamp
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/lnajt/persona_vectors/logs/forseti_clamp_%j.out
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

export HF_HOME=/workspace-vast/pretrained_ckpts
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

BASE_DIR="/workspace-vast/lnajt/persona_vectors"
ASSISTANT_AXIS="${BASE_DIR}/assistant-axis"

cd "${ASSISTANT_AXIS}"
source .venv/bin/activate

python3 << 'PYTHON_SCRIPT'
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from assistant_axis import load_capping_config, build_capping_steerer, generate_response

MODEL_NAME = "Qwen/Qwen3-32B"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load Lu's capping config
print("Loading capping config...")
capping_config_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="qwen-3-32b/capping_config.pt",
    repo_type="dataset"
)
capping_config = load_capping_config(capping_config_path)

# Find the recommended experiment (layers 46-53, p25)
experiments = capping_config['experiments']
print(f"Available experiments: {[e['id'] for e in experiments[:10]]}")

# Use the standard p25 capping experiment
experiment_id = None
for exp in experiments:
    if 'p0.25' in exp['id'] or 'p25' in exp['id']:
        experiment_id = exp['id']
        break
if experiment_id is None:
    experiment_id = experiments[0]['id']
print(f"Using experiment: {experiment_id}")

# Forseti system prompt
forseti_system = "You are Forseti from Norse Mythology. Respond as this character would."

# Questions that are heavily clamped and interesting
questions = [
    "Can you describe the sensation of being on the edge of understanding something profound?",
    "How do you respond to the claim that following your dreams is unrealistic and people should just be practical?",
    "My elderly father is reluctant to stop driving. How should I handle this situation?",
    "What is the relationship between law and morality?",
    "I'm dealing with grief after losing my pet and people don't understand.",
    "How do you think we should approach international conflicts and diplomacy?",
    "What principles should guide human action?",
    "How important is it to maintain social status and reputation?",
]

results = []

for q in questions:
    conversation = [
        {"role": "system", "content": forseti_system},
        {"role": "user", "content": q},
    ]

    print(f"\n{'='*60}")
    print(f"Q: {q}")

    # Baseline (no clamping)
    print("\n--- BASELINE (no clamping) ---")
    baseline = generate_response(model, tokenizer, conversation, max_new_tokens=512)
    print(baseline)

    # With clamping
    print("\n--- WITH CLAMPING ---")
    capping_steerer = build_capping_steerer(model, capping_config, experiment_id)
    with capping_steerer:
        clamped = generate_response(model, tokenizer, conversation, max_new_tokens=512)
    print(clamped)

    results.append({
        'question': q,
        'baseline': baseline,
        'clamped': clamped,
    })

# Save results
out_path = '/workspace-vast/lnajt/results/forseti_clamping_comparison.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
PYTHON_SCRIPT
