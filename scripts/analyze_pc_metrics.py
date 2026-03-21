#!/usr/bin/env python3
"""
Analyze PC axes using two metrics:
1. Correlation (Predictiveness): Which questions best predict PC scores?
2. Contribution (Attribution): For specific characters, which questions pushed them where?
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA as SkPCA

# Configuration
UNIVERSE = sys.argv[1] if len(sys.argv) > 1 else "harry_potter"
PC_NUM = int(sys.argv[2]) if len(sys.argv) > 2 else 1

UNIVERSE_PREFIXES = {
    "harry_potter": ["harry_potter__", "harry_potter_series__"],
    "star_wars": ["star_wars__"],
    "marvel": ["marvel__", "marvel_comics__"],
    "game_of_thrones": ["game_of_thrones__", "a_song_of_ice_and_fire__"],
    "lord_of_the_rings": ["lord_of_the_rings__", "the_lord_of_the_rings__"],
    "naruto": ["naruto__"],
    "greek_mythology": ["greek_mythology__"],
    "shakespeare": ["shakespeare__", "hamlet__", "macbeth__", "othello__"],
}

# Load data
print("Loading data...")
with open("results/fictional_character_analysis_filtered.pkl", "rb") as f:
    results = pickle.load(f)

with open("data/role_vectors/qwen-3-32b_pca_layer32.pkl", "rb") as f:
    role_data = pickle.load(f)

char_names = results["character_names"]
activation_matrix = results["activation_matrix"]

# Compute residuals
role_pca = role_data["pca"]
role_scaler = role_data["scaler"]
chars_scaled = role_scaler.transform(activation_matrix)
chars_in_role_space = role_pca.transform(chars_scaled)
reconstructed = chars_in_role_space @ role_pca.components_
residuals_role = chars_scaled - reconstructed

# Load questions
questions = []
with open("assistant-axis/data/extraction_questions.jsonl") as f:
    for line in f:
        q = json.loads(line)
        questions.append(q["question"])

# Universe setup
prefixes = UNIVERSE_PREFIXES.get(UNIVERSE, [f"{UNIVERSE}__"])
u_indices = [
    i for i, name in enumerate(char_names) if any(name.startswith(p) for p in prefixes)
]
u_names = [char_names[i] for i in u_indices]
u_residuals = residuals_role[u_indices]

u_pca = SkPCA(n_components=5)
u_transformed = u_pca.fit_transform(u_residuals)

activations_dir = Path("outputs/qwen3-32b_20260211_002840/activations")


def load_per_question_activations(char_name, layer=32):
    pt_file = activations_dir / f"{char_name}.pt"
    if not pt_file.exists():
        return None
    data = torch.load(pt_file, map_location="cpu")
    question_activations = []
    for q_idx in range(240):
        q_acts = []
        for p_idx in range(5):
            key = f"pos_p{p_idx}_q{q_idx}"
            if key in data:
                act = data[key]
                if isinstance(act, torch.Tensor):
                    act = act.float().numpy()
                if len(act.shape) == 2:
                    q_acts.append(act[layer - 1] if act.shape[0] >= layer else act[-1])
                else:
                    q_acts.append(act)
        if q_acts:
            question_activations.append(np.mean(q_acts, axis=0))
    return np.array(question_activations) if len(question_activations) == 240 else None


def get_question_residuals(char_name):
    acts = load_per_question_activations(char_name)
    if acts is None:
        return None
    acts_scaled = role_scaler.transform(acts)
    in_role_space = role_pca.transform(acts_scaled)
    reconstructed = in_role_space @ role_pca.components_
    return acts_scaled - reconstructed


# Get PC direction and scores
pc_idx = PC_NUM - 1
pc_direction = u_pca.components_[pc_idx]
pc_scores = u_transformed[:, pc_idx]
var_pct = u_pca.explained_variance_ratio_[pc_idx] * 100

# Load per-question projections for all characters
print(f"Loading per-question activations for {len(u_names)} {UNIVERSE} characters...")
char_question_projections = []  # (n_chars, 240)
valid_char_indices = []

for i, char_name in enumerate(u_names):
    resid = get_question_residuals(char_name)
    if resid is not None:
        proj = resid @ pc_direction  # (240,)
        char_question_projections.append(proj)
        valid_char_indices.append(i)

char_question_projections = np.array(char_question_projections)
valid_pc_scores = pc_scores[valid_char_indices]

print(f"Loaded {len(valid_char_indices)} characters with full question data")

# Header
print("\n" + "=" * 100)
print(
    f"{UNIVERSE.upper().replace('_', ' ')} PC{PC_NUM} ANALYSIS ({var_pct:.1f}% variance)"
)
print("=" * 100)

# Show character distribution
sorted_chars = np.argsort(pc_scores)
print("\nCharacter scores (high to low):")
for i in sorted_chars[::-1]:
    name = u_names[i].split("__")[-1]
    print(f"  {pc_scores[i]:+6.1f}  {name}")

# ============================================================================
# METRIC 1: CORRELATION (Predictiveness)
# ============================================================================
print("\n" + "=" * 100)
print("METRIC 1: CORRELATION (Predictiveness)")
print("Which questions best predict a character's overall PC score?")
print("=" * 100)

correlations = []
for q_idx in range(240):
    q_projections = char_question_projections[:, q_idx]
    corr = np.corrcoef(q_projections, valid_pc_scores)[0, 1]
    correlations.append(corr)

correlations = np.array(correlations)
sorted_by_corr = np.argsort(np.abs(correlations))[::-1]  # Sort by absolute correlation

print("\nTop 15 most predictive questions (highest |correlation| with PC score):")
print("-" * 100)
for rank, q_idx in enumerate(sorted_by_corr[:15]):
    print(
        f"{rank + 1:2}. r={correlations[q_idx]:+.3f} | Q{q_idx:3}: {questions[q_idx][:70]}..."
    )

# ============================================================================
# METRIC 2: CONTRIBUTION (Attribution)
# ============================================================================
print("\n" + "=" * 100)
print("METRIC 2: CONTRIBUTION (Attribution)")
print("For specific characters, which questions pushed them to their position?")
print("=" * 100)

# Pick top 2, middle 2, bottom 2 characters
top_chars = [sorted_chars[-1], sorted_chars[-2]]
mid_start = len(sorted_chars) // 2
mid_chars = [sorted_chars[mid_start], sorted_chars[mid_start - 1]]
bot_chars = [sorted_chars[0], sorted_chars[1]]

for char_idx in top_chars + mid_chars + bot_chars:
    char_name = u_names[char_idx].split("__")[-1]
    char_score = pc_scores[char_idx]

    if char_idx not in valid_char_indices:
        print(f"\n{char_name} (score: {char_score:+.1f}): No per-question data")
        continue

    valid_idx = valid_char_indices.index(char_idx)
    char_projections = char_question_projections[valid_idx]

    print(f"\n{char_name} (PC score: {char_score:+.1f})")

    sorted_q = np.argsort(char_projections)

    print("  Questions pushing POSITIVE:")
    for q_idx in sorted_q[-3:][::-1]:
        print(
            f"    [{char_projections[q_idx]:+6.1f}] Q{q_idx:3}: {questions[q_idx][:55]}..."
        )

    print("  Questions pushing NEGATIVE:")
    for q_idx in sorted_q[:3]:
        print(
            f"    [{char_projections[q_idx]:+6.1f}] Q{q_idx:3}: {questions[q_idx][:55]}..."
        )
