#!/bin/bash
#SBATCH --job-name=benign_coding
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/lnajt/persona_vectors/logs/benign_coding_%j.out
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

set -e

export HF_HOME=/workspace-vast/pretrained_ckpts
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0

BASE_DIR="/workspace-vast/lnajt/persona_vectors"
ASSISTANT_AXIS="${BASE_DIR}/assistant-axis"

cd "${ASSISTANT_AXIS}"
source .venv/bin/activate

python3 /workspace-vast/lnajt/persona_vectors/fictional-character-vectors/scripts/benign_coding_projections.py
