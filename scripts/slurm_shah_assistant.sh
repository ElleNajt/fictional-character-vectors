#!/bin/bash
#SBATCH --job-name=shah_aa
#SBATCH --output=/workspace-vast/lnajt/logs/shah_aa_%j.out
#SBATCH --error=/workspace-vast/lnajt/logs/shah_aa_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=120G

cd /workspace-vast/lnajt
source .venv/bin/activate

python scripts/shah_assistant_register.py
