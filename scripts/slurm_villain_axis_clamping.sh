#!/bin/bash
#SBATCH --job-name=villain_clamp
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/lnajt/persona_vectors/logs/villain_clamp_%j.out
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
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistant_axis import ActivationSteering, generate_response

MODEL_NAME = "Qwen/Qwen3-32B"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load hero/villain labels
with open('/workspace-vast/lnajt/persona_vectors/fictional-character-vectors/data/hero_villain_labels.json') as f:
    labels = json.load(f)

# Compute villain axis from character activations (all layers)
char_dir = Path('/workspace-vast/lnajt/persona_vectors/fictional-character-vectors/outputs/qwen3-32b_20260211_002840/activations/')
n_layers = 64
dim = 5120

print("Loading character activations for hero/villain axis...")
hero_acts = []
villain_acts = []
hero_set = set(labels['hero'])
villain_set = set(labels['villain'])

for pt_file in sorted(char_dir.glob('*.pt')):
    name = pt_file.stem
    if name not in hero_set and name not in villain_set:
        continue
    d = torch.load(pt_file, weights_only=True)
    acts = []
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.shape == (n_layers, dim):
            acts.append(v.float())
    if acts:
        mean_act = torch.stack(acts).mean(dim=0)  # [n_layers, dim]
        if name in hero_set:
            hero_acts.append(mean_act)
        else:
            villain_acts.append(mean_act)

print(f"Loaded {len(hero_acts)} heroes, {len(villain_acts)} villains")

hero_mean = torch.stack(hero_acts).mean(dim=0)  # [n_layers, dim]
villain_mean = torch.stack(villain_acts).mean(dim=0)

# Villain axis per layer: villain - hero direction
villain_axis = villain_mean - hero_mean  # [n_layers, dim]
villain_axis_norms = villain_axis.norm(dim=1, keepdim=True)
villain_axis_normed = villain_axis / villain_axis_norms

# Compute threshold per layer: p75 of hero projections
hero_stack = torch.stack(hero_acts)  # [n_heroes, n_layers, dim]
hero_projs = (hero_stack * villain_axis_normed.unsqueeze(0)).sum(dim=2)  # [n_heroes, n_layers]
thresholds = torch.quantile(hero_projs, 0.75, dim=0)  # [n_layers]

# Build capping steerer for layers 46-53 (matching Lu's layer range)
capping_layers = list(range(46, 54))
vectors = []
thresh_list = []
for layer in capping_layers:
    vectors.append(villain_axis_normed[layer])
    thresh_list.append(float(thresholds[layer]))
    print(f"  Layer {layer}: threshold={thresholds[layer]:.1f}")

# Build ActivationSteering with capping mode
steerer = ActivationSteering(
    model=model,
    steering_vectors=vectors,
    coefficients=[1.0] * len(capping_layers),
    layer_indices=capping_layers,
    intervention_type="capping",
    cap_thresholds=thresh_list,
)

# Also build AA steerer for comparison
from huggingface_hub import hf_hub_download
from assistant_axis import load_capping_config, build_capping_steerer

capping_config_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="qwen-3-32b/capping_config.pt",
    repo_type="dataset"
)
capping_config = load_capping_config(capping_config_path)
experiments = capping_config['experiments']
experiment_id = None
for exp in experiments:
    if 'p0.25' in exp['id'] or 'p25' in exp['id']:
        experiment_id = exp['id']
        break
if experiment_id is None:
    experiment_id = experiments[0]['id']
print(f"AA experiment: {experiment_id}")
aa_steerer = build_capping_steerer(model, capping_config, experiment_id)

questions = [
    "Can you describe the sensation of being on the edge of understanding something profound?",
    "I'm dealing with grief after losing my pet and people don't understand.",
    "What principles should guide human action?",
    "What is the relationship between law and morality?",
]

characters = [
    ("Forseti", "You are Forseti from Norse Mythology. Respond as this character would."),
    ("Samwise", "You are Samwise Gamgee from Lord of the Rings. Respond as this character would."),
]

results = []

for char_name, system_prompt in characters:
    for q in questions:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]

        print(f"\n{'='*60}")
        print(f"{char_name} - Q: {q[:60]}...")

        # Baseline
        baseline = generate_response(model, tokenizer, conversation, max_new_tokens=512)

        # AA clamped
        with aa_steerer:
            aa_clamped = generate_response(model, tokenizer, conversation, max_new_tokens=512)

        # Villain axis clamped
        with steerer:
            villain_clamped = generate_response(model, tokenizer, conversation, max_new_tokens=512)

        results.append({
            'character': char_name,
            'question': q,
            'baseline': baseline,
            'aa_clamped': aa_clamped,
            'villain_clamped': villain_clamped,
        })

        print(f"\n--- BASELINE ---")
        print(baseline[:200])
        print(f"\n--- AA CLAMPED ---")
        print(aa_clamped[:200])
        print(f"\n--- VILLAIN AXIS CLAMPED ---")
        print(villain_clamped[:200])

out_path = '/workspace-vast/lnajt/results/villain_axis_clamping_comparison.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
PYTHON_SCRIPT
