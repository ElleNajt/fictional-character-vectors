"""LLM-based response diffing for residual PC interpretation.

For each universe's residual PC1:
  - Take 5 highest and 5 lowest characters
  - Get top 100 most discriminative questions (by per-question variance)
  - Feed anonymized response pairs to an LLM
  - Ask it to describe systematic differences between Group A and Group B

Requires OPENROUTER_API_KEY environment variable (or pass via .env).
"""

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import requests
import torch
from sklearn.decomposition import PCA as SkPCA

# --- Config ---
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4"  # good balance of quality and cost
N_EXTREME = 5
N_QUESTIONS = 100  # top 100 by per-question variance
N_SAMPLE_RESPONSES = 10  # questions to include in the LLM prompt (subset of top 100)

# --- Load data ---
print("Loading data...")
with open("results/fictional_character_analysis_filtered.pkl", "rb") as f:
    char_data = pickle.load(f)

questions = []
with open("assistant-axis/data/extraction_questions.jsonl") as f:
    for line in f:
        questions.append(json.loads(line)["question"])

char_names = char_data["character_names"]
activation_matrix = char_data["activation_matrix"]

with open("data/role_vectors/qwen-3-32b_pca_layer32.pkl", "rb") as f:
    lu_data = pickle.load(f)
role_pca = lu_data["pca"]
scaler = lu_data["scaler"]

chars_scaled = scaler.transform(activation_matrix)
chars_in_role_space = role_pca.transform(chars_scaled)
reconstructed = chars_in_role_space @ role_pca.components_
residuals = chars_scaled - reconstructed

ALL_UNIVERSES = {
    "Harry Potter": ["harry_potter__", "harry_potter_series__"],
    "Star Wars": ["star_wars__"],
    "LOTR": ["lord_of_the_rings__"],
    "Marvel": ["marvel__", "marvel_comics__"],
    "Game of Thrones": ["game_of_thrones__"],
    "Naruto": ["naruto__"],
    "Greek Mythology": ["greek_mythology__"],
    "Chinese Mythology": ["chinese_mythology__"],
    "Hindu Mythology": ["hindu_mythology__"],
    "Norse Mythology": ["norse_mythology__"],
    "Egyptian Mythology": ["egyptian_mythology__"],
    "Shakespeare": ["shakespeare__"],
}

responses_dir = Path(
    "persona_vectors/fictional-character-vectors/outputs/qwen3-32b_20260211_002840/responses"
)
activations_dir = Path("outputs/qwen3-32b_20260211_002840/activations")


def get_universe_indices(prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def load_responses(char_name):
    """Load responses for a character. Returns dict: question_index -> list of response texts."""
    fpath = responses_dir / f"{char_name}.jsonl"
    if not fpath.exists():
        return None
    responses = {}
    with open(fpath) as f:
        for line in f:
            data = json.loads(line)
            qi = data["question_index"]
            text = data["conversation"][-1]["content"]
            if qi not in responses:
                responses[qi] = []
            responses[qi].append(text)
    return responses


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
                    q_acts.append(act[layer - 1])
                else:
                    q_acts.append(act)
        if q_acts:
            question_activations.append(np.mean(q_acts, axis=0))
    return np.array(question_activations) if question_activations else None


def call_llm(prompt, system="You are a careful linguistic analyst."):
    """Call OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.0,
    }
    resp = requests.post(API_URL, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# --- Main ---
# Load existing results to skip already-completed universes
existing_results_path = Path("results/llm_response_diff.txt")
existing_text = (
    existing_results_path.read_text() if existing_results_path.exists() else ""
)
out = [existing_text.rstrip()] if existing_text.strip() else []

for universe, prefixes in ALL_UNIVERSES.items():
    if (
        f"# {universe}\n" in existing_text
        and "LLM Analysis:"
        in existing_text.split(f"# {universe}\n")[-1].split("######")[0]
    ):
        print(f"Skipping {universe} (already has LLM results)")
        continue
    indices = get_universe_indices(prefixes)
    if len(indices) < 20:
        continue

    u_residuals = residuals[indices]
    u_names = [char_names[i] for i in indices]

    u_pca = SkPCA(n_components=1)
    scores = u_pca.fit_transform(u_residuals)[:, 0]
    pc1_dir = u_pca.components_[0]

    sorted_idx = np.argsort(scores)
    high_chars = [u_names[i] for i in sorted_idx[-N_EXTREME:][::-1]]
    low_chars = [u_names[i] for i in sorted_idx[:N_EXTREME]]

    header = f"\n{'#' * 70}\n# {universe}\n{'#' * 70}"
    high_pretty = [n.split("__")[-1].replace("_", " ").title() for n in high_chars]
    low_pretty = [n.split("__")[-1].replace("_", " ").title() for n in low_chars]
    header += f"\n  HIGH: {', '.join(high_pretty)}"
    header += f"\n  LOW:  {', '.join(low_pretty)}"
    out.append(header)
    print(header)

    # Find top N_QUESTIONS by per-question variance along PC1
    # Fall back to random questions if per-question activations aren't available
    all_projections = []
    valid_names = []
    for cname in u_names:
        acts = load_per_question_activations(cname)
        if acts is not None and len(acts) == 240:
            acts_scaled = scaler.transform(acts)
            in_role = role_pca.transform(acts_scaled)
            acts_resid = acts_scaled - in_role @ role_pca.components_
            projections = acts_resid @ pc1_dir
            all_projections.append(projections)
            valid_names.append(cname)

    if all_projections:
        all_projections = np.array(all_projections)
        question_variance = np.var(all_projections, axis=0)
        top_q_indices = np.argsort(question_variance)[::-1][:N_QUESTIONS]
    else:
        # No per-question activations: use first N_QUESTIONS questions
        print("  (no per-question activations, using default question order)")
        top_q_indices = np.arange(N_QUESTIONS)

    # Load responses for extreme characters
    high_responses = {}
    low_responses = {}
    for name in high_chars:
        resp = load_responses(name)
        if resp:
            high_responses[name] = resp
    for name in low_chars:
        resp = load_responses(name)
        if resp:
            low_responses[name] = resp

    if not high_responses or not low_responses:
        out.append("  (missing response data)")
        print("  (missing response data)")
        continue

    # Build the LLM prompt with N_SAMPLE_RESPONSES sampled from top 100
    # Pick questions where both groups have data
    usable_qs = []
    for qi in top_q_indices:
        has_high = any(qi in r for r in high_responses.values())
        has_low = any(qi in r for r in low_responses.values())
        if has_high and has_low:
            usable_qs.append(int(qi))
    sample_qs = usable_qs[:N_SAMPLE_RESPONSES]

    prompt_parts = [
        "Below are responses from two groups of characters (Group A and Group B) to the same questions.",
        "Each group contains 5 characters from the same fictional universe.",
        "For each question, I show one response from a Group A character and one from a Group B character.",
        "",
        "Please analyze the systematic differences between Group A and Group B responses.",
        "Focus on: communication style, perspective/worldview, how they engage with the questioner,",
        "level of abstraction vs concreteness, emotional tone, and any other patterns you notice.",
        "Be specific and cite examples from the responses.",
        "",
    ]

    for i, qi in enumerate(sample_qs):
        q_text = questions[qi] if qi < len(questions) else f"question {qi}"
        prompt_parts.append(f"--- Question {i + 1}: {q_text} ---")

        # Pick a random HIGH character that has this question
        for name, resp in high_responses.items():
            if qi in resp:
                text = resp[qi][0][:1500]  # truncate long responses
                prompt_parts.append(f"\nGroup A response:\n{text}")
                break

        for name, resp in low_responses.items():
            if qi in resp:
                text = resp[qi][0][:1500]
                prompt_parts.append(f"\nGroup B response:\n{text}")
                break

        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    print(f"  Calling LLM ({len(sample_qs)} questions, {len(prompt)} chars)...")
    try:
        response = call_llm(prompt)
        out.append(f"\n  LLM Analysis:\n{response}")
        print(f"  Response received ({len(response)} chars)")
    except Exception as e:
        out.append(f"\n  LLM Error: {e}")
        print(f"  Error: {e}")

    # Rate limit
    time.sleep(2)

Path("results/llm_response_diff.txt").write_text("\n".join(out))
print(f"\nSaved results/llm_response_diff.txt")
