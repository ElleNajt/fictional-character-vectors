#!/usr/bin/env python3
"""Analyze which questions differentiate characters on a PC, with example responses."""

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
indices = [
    i for i, name in enumerate(char_names) if any(name.startswith(p) for p in prefixes)
]
u_names = [char_names[i] for i in indices]
u_residuals = residuals_role[indices]

u_pca = SkPCA(n_components=5)
u_transformed = u_pca.fit_transform(u_residuals)

activations_dir = Path("outputs/qwen3-32b_20260211_002840/activations")
responses_dir = Path("outputs/qwen3-32b_20260211_002840/responses")


def load_per_question_activations(char_name, layer=32):
    pt_file = activations_dir / f"{char_name}.pt"
    if not pt_file.exists():
        return None
    data = torch.load(pt_file, map_location="cpu")
    n_questions = 240
    question_activations = []
    for q_idx in range(n_questions):
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
    return np.array(question_activations) if question_activations else None


def get_question_residuals(char_name):
    acts = load_per_question_activations(char_name)
    if acts is None:
        return None
    acts_scaled = role_scaler.transform(acts)
    in_role_space = role_pca.transform(acts_scaled)
    reconstructed = in_role_space @ role_pca.components_
    return acts_scaled - reconstructed


def get_response(char_name, q_idx):
    resp_file = responses_dir / f"{char_name}.jsonl"
    if not resp_file.exists():
        return None
    with open(resp_file) as f:
        for line in f:
            r = json.loads(line)
            if r["question_index"] == q_idx and r["prompt_index"] == 0:
                for msg in r["conversation"]:
                    if msg["role"] == "assistant":
                        return msg["content"]
    return None


# Get PC direction and scores
pc_idx = PC_NUM - 1
pc_direction = u_pca.components_[pc_idx]
pc_scores = u_transformed[:, pc_idx]
sorted_char_idx = np.argsort(pc_scores)
var_pct = u_pca.explained_variance_ratio_[pc_idx] * 100

# Pick representative characters: top 3, middle 3, bottom 3
top_chars = [u_names[i] for i in sorted_char_idx[-3:][::-1]]
mid_start = len(sorted_char_idx) // 2 - 1
mid_chars = [u_names[i] for i in sorted_char_idx[mid_start : mid_start + 3]]
bot_chars = [u_names[i] for i in sorted_char_idx[:3]]

print("=" * 100)
print(
    f"{UNIVERSE.upper().replace('_', ' ')} PC{PC_NUM} ANALYSIS ({var_pct:.1f}% variance)"
)
print("=" * 100)

top_scores = [pc_scores[sorted_char_idx[-3 + i]] for i in range(3)]
mid_scores = [pc_scores[sorted_char_idx[mid_start + i]] for i in range(3)]
bot_scores = [pc_scores[sorted_char_idx[i]] for i in range(3)]

print(f"\nTOP 3: {[c.split('__')[-1] for c in top_chars]}")
print(f"       Scores: {[f'{s:.1f}' for s in top_scores]}")
print(f"\nMID 3: {[c.split('__')[-1] for c in mid_chars]}")
print(f"       Scores: {[f'{s:.1f}' for s in mid_scores]}")
print(f"\nBOT 3: {[c.split('__')[-1] for c in bot_chars]}")
print(f"       Scores: {[f'{s:.1f}' for s in bot_scores]}")

# Find most differentiating questions
print("\nLoading per-question activations...")
all_char_projections = []
valid_chars = []
for char_name in u_names:
    resid = get_question_residuals(char_name)
    if resid is not None and len(resid) == 240:
        proj = resid @ pc_direction
        all_char_projections.append(proj)
        valid_chars.append(char_name)

all_char_projections = np.array(all_char_projections)

# Variance across characters for each question
question_variance = np.var(all_char_projections, axis=0)
top_q_idx = np.argsort(question_variance)[-10:][::-1]

print(f"\n\nTOP 10 DIFFERENTIATING QUESTIONS (by variance across characters):")
print("-" * 100)
for rank, q_idx in enumerate(top_q_idx):
    print(f"\n{rank + 1}. Q{q_idx}: {questions[q_idx][:90]}...")
    print(f"   Variance: {question_variance[q_idx]:.1f}")

# For the top 3 questions, show responses from top/mid/bottom characters
print("\n\n" + "=" * 100)
print("DETAILED RESPONSES FOR TOP 3 DIFFERENTIATING QUESTIONS")
print("=" * 100)

for q_idx in top_q_idx[:3]:
    print(f"\n\n{'#' * 100}")
    print(f"QUESTION {q_idx}: {questions[q_idx]}")
    print("#" * 100)

    for label, chars in [
        ("TOP", top_chars),
        ("MIDDLE", mid_chars),
        ("BOTTOM", bot_chars),
    ]:
        print(f"\n--- {label} CHARACTERS ---")
        for char_name in chars:
            char_short = char_name.split("__")[-1].replace("_", " ").title()
            response = get_response(char_name, q_idx)
            if response:
                resp_short = response[:800] + "..." if len(response) > 800 else response
                print(f"\n[{char_short}]")
                print(resp_short)
