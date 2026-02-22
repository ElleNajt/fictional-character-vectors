#!/bin/bash
#SBATCH --job-name=steering-v2
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/lnajt/persona_vectors/fictional-character-vectors/results/steering_v2_%j.out

cd /workspace-vast/lnajt/persona_vectors/fictional-character-vectors

HF_HOME=/workspace-vast/annas/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python3 src/analysis/steering_experiment.py
