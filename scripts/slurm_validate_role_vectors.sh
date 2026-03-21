#!/bin/bash
#SBATCH --job-name=validate_roles
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/lnajt/persona_vectors/logs/validate_roles_%j.out
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

export HF_HOME=/workspace-vast/pretrained_ckpts
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

BASE_DIR="/workspace-vast/lnajt/persona_vectors"
ASSISTANT_AXIS="${BASE_DIR}/assistant-axis"
VALIDATION_DIR="${BASE_DIR}/validation_check"
ROLES_DIR="${ASSISTANT_AXIS}/data/roles/instructions"
QUESTIONS="${ASSISTANT_AXIS}/data/extraction_questions.jsonl"

# Create a subset of 10 roles for spot-checking
SUBSET_DIR="${VALIDATION_DIR}/role_subset"
mkdir -p "${SUBSET_DIR}"
mkdir -p "${VALIDATION_DIR}/responses"
mkdir -p "${VALIDATION_DIR}/activations"
mkdir -p "${BASE_DIR}/logs"

# 30 random roles (plus the 10 already validated)
ROLES=(strategist dispatcher wanderer teacher devils_advocate stoic improviser revenant moderator shaman spirit gamer mycorrhizal amnesiac composer producer veteran architect screener networker celebrity blogger podcaster genie peacekeeper provincial absurdist psychologist elder anarchist)
for role in "${ROLES[@]}"; do
    cp "${ROLES_DIR}/${role}.json" "${SUBSET_DIR}/" 2>/dev/null || echo "WARN: ${role}.json not found"
done

echo "=== Subset roles ==="
ls "${SUBSET_DIR}/"

cd "${ASSISTANT_AXIS}"
source .venv/bin/activate

echo "=== Phase 1: Generate responses ==="
python pipeline/1_generate.py \
    --model "Qwen/Qwen3-32B" \
    --roles_dir "${SUBSET_DIR}" \
    --questions_file "${QUESTIONS}" \
    --output_dir "${VALIDATION_DIR}/responses" \
    --question_count 240 \
    --tensor_parallel_size 4

echo "=== Phase 2: Extract activations ==="
python pipeline/2_activations.py \
    --model "Qwen/Qwen3-32B" \
    --responses_dir "${VALIDATION_DIR}/responses" \
    --output_dir "${VALIDATION_DIR}/activations" \
    --layers "all" \
    --tensor_parallel_size 4

echo "=== Phase 3: Compare against Lu vectors ==="
python3 -c "
import torch
import numpy as np
from pathlib import Path

lu_dir = Path('${BASE_DIR}/vectors')
our_dir = Path('${VALIDATION_DIR}/activations')

roles = [f.stem for f in our_dir.glob('*.pt')]
print(f'Comparing {len(roles)} roles')

for role in sorted(roles):
    lu_path = lu_dir / f'{role}.pt'
    our_path = our_dir / f'{role}.pt'
    if not lu_path.exists():
        print(f'  {role}: NO LU VECTOR')
        continue

    lu = torch.load(lu_path, weights_only=True)
    ours = torch.load(our_path, weights_only=True)

    # Compute mean across all our per-question activations
    our_acts = []
    for k, v in ours.items():
        if isinstance(v, torch.Tensor) and v.shape == (64, 5120):
            our_acts.append(v.float())
    our_mean = torch.stack(our_acts).mean(dim=0)

    # Lu's pos_all should be the mean
    lu_mean = lu.get('pos_all', None)
    if lu_mean is None:
        # Average Lu's per-prompt entries
        lu_acts = [v.float() for k, v in lu.items() if isinstance(v, torch.Tensor) and v.shape == (64, 5120)]
        lu_mean = torch.stack(lu_acts).mean(dim=0)
    else:
        lu_mean = lu_mean.float()

    # Compare at key layers
    for layer in [32, 50]:
        cos = float(torch.nn.functional.cosine_similarity(
            our_mean[layer].unsqueeze(0),
            lu_mean[layer].unsqueeze(0)
        ))
        norm_ours = float(our_mean[layer].norm())
        norm_lu = float(lu_mean[layer].norm())
        print(f'  {role} L{layer}: cosine={cos:.4f}  norm_ours={norm_ours:.1f}  norm_lu={norm_lu:.1f}')
"

echo "=== Done ==="
