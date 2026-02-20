"""LLM-based feature coding for residual PC interpretation.

Two-phase pipeline:
  Phase 1 (discover): For each universe × PC, show extreme characters' responses
    to an LLM and ask it to propose distinguishing features.
  Phase 2 (code): For each character in universe, show responses and ask LLM
    to rate on the discovered features.

Requires OPENROUTER_API_KEY environment variable.

Usage:
    python src/analysis/llm_feature_coding.py discover   # Phase 1
    python src/analysis/llm_feature_coding.py code        # Phase 2 (needs Phase 1 output)
    python src/analysis/llm_feature_coding.py all         # Both phases
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from sklearn.decomposition import PCA as SkPCA

# --- Config ---
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DISCOVER_MODEL = "anthropic/claude-sonnet-4"
CODE_MODEL = "anthropic/claude-sonnet-4"
N_DISCOVER_PER_SIDE = 5  # odd-ranked: 1st, 3rd, 5th, 7th, 9th from each end
N_CODE_PER_SIDE = 10  # even-ranked: 2nd, 4th, ..., 20th from each end
N_DISCOVER_QUESTIONS = 10  # questions shown in discovery prompt
N_CODE_QUESTIONS = 10  # questions shown when coding each character
N_DISCRIMINATIVE_POOL = 50  # top 50 most discriminative questions to sample from
N_PCS = 2  # code features for PC1 and PC2

# --- Paths ---
ACTIVATIONS_DIR = Path("outputs/qwen3-32b_20260211_002840/activations")
RESPONSES_DIR = Path("outputs/qwen3-32b_20260211_002840/responses")
CHAR_DATA_PATH = "results/fictional_character_analysis_filtered.pkl"
LU_PCA_PATH = "data/role_vectors/qwen-3-32b_pca_layer32.pkl"
QUESTIONS_PATH = "assistant-axis/data/extraction_questions.jsonl"
SCHEMA_PATH = "results/llm_feature_schemas.json"
CODED_PATH = "results/llm_feature_coded.json"

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


def load_data():
    with open(CHAR_DATA_PATH, "rb") as f:
        char_data = pickle.load(f)
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            questions.append(json.loads(line)["question"])
    with open(LU_PCA_PATH, "rb") as f:
        lu_data = pickle.load(f)
    return char_data, questions, lu_data


def get_universe_indices(char_names, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def load_per_question_activations(char_name, layer=32):
    pt_file = ACTIVATIONS_DIR / f"{char_name}.pt"
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


def load_responses(char_name):
    fpath = RESPONSES_DIR / f"{char_name}.jsonl"
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


def call_llm(prompt, system="You are a careful linguistic analyst.", model=None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model or CODE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.0,
    }
    resp = requests.post(API_URL, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def strided_select(sorted_idx, n, stride_offset):
    """Select characters from a sorted index using striding.

    sorted_idx: array sorted by PC score (lowest first)
    stride_offset: 0 for odd-ranked (1st, 3rd, ...), 1 for even-ranked (2nd, 4th, ...)
    Returns (high_chars_indices, low_chars_indices) from top and bottom of sorted_idx.
    """
    # From the high end: sorted_idx[-1] is rank 1, sorted_idx[-2] is rank 2, etc.
    high_indices = [sorted_idx[-(i + 1)] for i in range(stride_offset, 2 * n, 2)][:n]
    # From the low end: sorted_idx[0] is rank 1, sorted_idx[1] is rank 2, etc.
    low_indices = [sorted_idx[i] for i in range(stride_offset, 2 * n, 2)][:n]
    return high_indices, low_indices


def get_discriminative_questions(u_names, pc_dir, scaler, role_pca):
    """Find questions with highest per-question variance along a PC direction."""
    all_projections = []
    for cname in u_names:
        acts = load_per_question_activations(cname)
        if acts is not None and len(acts) == 240:
            acts_scaled = scaler.transform(acts)
            in_role = role_pca.transform(acts_scaled)
            acts_resid = acts_scaled - in_role @ role_pca.components_
            all_projections.append(acts_resid @ pc_dir)

    if not all_projections:
        return list(range(N_DISCOVER_QUESTIONS)), None

    all_proj = np.array(all_projections)
    question_variance = np.var(all_proj, axis=0)
    top_q_indices = np.argsort(question_variance)[::-1].tolist()
    return top_q_indices, question_variance


def get_responses_for_questions(char_name, q_indices):
    """Load responses for specific questions. Returns list of (q_idx, text)."""
    resp = load_responses(char_name)
    if not resp:
        return []
    pairs = []
    for qi in q_indices:
        if qi in resp:
            pairs.append((qi, resp[qi][0][:800]))  # truncate
    return pairs


# ============================================================
# Phase 1: Feature Discovery
# ============================================================

DISCOVER_PROMPT_TEMPLATE = """Below are responses from two groups of anonymous characters to the same questions.
Group A contains 5 characters; Group B contains 5 characters. They are from the same fictional universe.

{response_pairs}

Based ONLY on the response text above (not any outside knowledge), propose exactly 6 features that
systematically distinguish Group A from Group B. Each feature should be:

1. Observable purely from the response text (not requiring knowledge of who the characters are)
2. A spectrum (not binary) -- something you could rate on a 1-5 scale
3. About HOW the character communicates, not WHAT they're talking about

For each feature, provide:
- A short name (2-4 words)
- A description of the 1 end (low) and 5 end (high)
- Which group (A or B) tends to score higher

Reply in this exact JSON format:
```json
[
  {{
    "name": "Feature Name",
    "low_description": "What a score of 1 looks like",
    "high_description": "What a score of 5 looks like",
    "high_group": "A or B"
  }},
  ...
]
```

Return ONLY the JSON array, no other text."""


def build_discovery_prompt(high_chars, low_chars, q_indices, questions):
    """Build prompt showing interleaved responses from high/low groups."""
    parts = []
    for i, qi in enumerate(q_indices[:N_DISCOVER_QUESTIONS]):
        q_text = questions[qi]
        parts.append(f"--- Question {i + 1}: {q_text} ---")

        # Pick a high char that has this question
        for name in high_chars:
            resp_pairs = get_responses_for_questions(name, [qi])
            if resp_pairs:
                parts.append(f"\nGroup A response:\n{resp_pairs[0][1]}")
                break

        for name in low_chars:
            resp_pairs = get_responses_for_questions(name, [qi])
            if resp_pairs:
                parts.append(f"\nGroup B response:\n{resp_pairs[0][1]}")
                break

        parts.append("")

    return DISCOVER_PROMPT_TEMPLATE.format(response_pairs="\n".join(parts))


def phase_discover(char_data, questions, lu_data):
    print("=== Phase 1: Feature Discovery ===\n")

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = lu_data["pca"]
    scaler = lu_data["scaler"]

    chars_scaled = scaler.transform(activation_matrix)
    chars_in_role_space = role_pca.transform(chars_scaled)
    reconstructed = chars_in_role_space @ role_pca.components_
    residuals = chars_scaled - reconstructed

    # Load existing schemas to allow resuming
    if Path(SCHEMA_PATH).exists():
        with open(SCHEMA_PATH) as f:
            schemas = json.load(f)
    else:
        schemas = {}

    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            continue

        u_residuals = residuals[indices]
        u_names = [char_names[i] for i in indices]

        u_pca = SkPCA(n_components=N_PCS)
        u_scores = u_pca.fit_transform(u_residuals)

        for pc_idx in range(N_PCS):
            key = f"{universe}__PC{pc_idx + 1}"
            if key in schemas:
                print(f"Skipping {key} (already discovered)")
                continue

            scores = u_scores[:, pc_idx]
            pc_dir = u_pca.components_[pc_idx]
            sorted_idx = np.argsort(scores)

            # Odd-ranked characters for discovery (1st, 3rd, 5th, ...)
            high_idx, low_idx = strided_select(
                sorted_idx, N_DISCOVER_PER_SIDE, stride_offset=0
            )
            high_chars = [u_names[i] for i in high_idx]
            low_chars = [u_names[i] for i in low_idx]

            pretty = lambda n: n.split("__")[-1].replace("_", " ").title()
            high_pretty = [pretty(n) for n in high_chars]
            low_pretty = [pretty(n) for n in low_chars]

            print(f"\n{key} (discovery, odd-ranked):")
            print(f"  HIGH: {', '.join(high_pretty)}")
            print(f"  LOW:  {', '.join(low_pretty)}")

            # Get discriminative questions for this PC
            top_q_indices, _ = get_discriminative_questions(
                u_names, pc_dir, scaler, role_pca
            )

            # Filter to questions where both groups have responses
            usable_qs = []
            for qi in top_q_indices:
                has_high = any(get_responses_for_questions(n, [qi]) for n in high_chars)
                has_low = any(get_responses_for_questions(n, [qi]) for n in low_chars)
                if has_high and has_low:
                    usable_qs.append(qi)
                if len(usable_qs) >= N_DISCOVER_QUESTIONS:
                    break

            prompt = build_discovery_prompt(high_chars, low_chars, usable_qs, questions)
            print(f"  Calling LLM for feature discovery ({len(prompt)} chars)...")

            try:
                response = call_llm(prompt, model=DISCOVER_MODEL)
                # Parse JSON from response
                json_str = response
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                features = json.loads(json_str.strip())

                schemas[key] = {
                    "universe": universe,
                    "pc": pc_idx + 1,
                    "high_chars": high_pretty,
                    "low_chars": low_pretty,
                    "features": features,
                    "discover_question_indices": usable_qs[:N_DISCOVER_QUESTIONS],
                }
                print(f"  Discovered {len(features)} features:")
                for f in features:
                    print(
                        f"    - {f['name']}: {f['low_description']} <-> {f['high_description']}"
                    )

            except Exception as e:
                print(f"  Error: {e}")
                schemas[key] = {"error": str(e)}

            # Save after each to allow resuming
            with open(SCHEMA_PATH, "w") as f:
                json.dump(schemas, f, indent=2, ensure_ascii=False)

            time.sleep(2)

    print(f"\nSchemas saved to {SCHEMA_PATH}")
    return schemas


# ============================================================
# Phase 2: Feature Coding
# ============================================================

CODE_PROMPT_TEMPLATE = """Below are responses from a single anonymous character to several questions.

{responses}

Rate this character on each of the following features, using a scale of 1-5.
Base your ratings ONLY on the text above.

Features:
{feature_descriptions}

Reply in this exact JSON format:
```json
{{
{json_keys}
}}
```

Return ONLY the JSON object, no other text."""


def build_code_prompt(char_name, q_indices, questions, features):
    """Build prompt to code one character on the discovered features."""
    resp_pairs = get_responses_for_questions(char_name, q_indices)
    if not resp_pairs:
        return None

    resp_parts = []
    for qi, text in resp_pairs[:N_CODE_QUESTIONS]:
        q_text = questions[qi]
        resp_parts.append(f"Q: {q_text}\nA: {text}\n")

    feature_descs = []
    json_keys = []
    for i, f in enumerate(features):
        feature_descs.append(
            f"{i + 1}. {f['name']}: 1 = {f['low_description']}, 5 = {f['high_description']}"
        )
        json_keys.append(f'  "{f["name"]}": <1-5>')

    return CODE_PROMPT_TEMPLATE.format(
        responses="\n".join(resp_parts),
        feature_descriptions="\n".join(feature_descs),
        json_keys=",\n".join(json_keys),
    )


def phase_code(char_data, questions, lu_data):
    print("=== Phase 2: Feature Coding ===\n")

    if not Path(SCHEMA_PATH).exists():
        print(f"Error: {SCHEMA_PATH} not found. Run 'discover' first.")
        return None

    with open(SCHEMA_PATH) as f:
        schemas = json.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = lu_data["pca"]
    scaler = lu_data["scaler"]

    chars_scaled = scaler.transform(activation_matrix)
    chars_in_role_space = role_pca.transform(chars_scaled)
    reconstructed = chars_in_role_space @ role_pca.components_
    residuals = chars_scaled - reconstructed

    # Load existing coded data to allow resuming
    if Path(CODED_PATH).exists():
        with open(CODED_PATH) as f:
            coded = json.load(f)
    else:
        coded = {}

    rng = np.random.RandomState(42)

    for schema_key, schema in schemas.items():
        if "error" in schema:
            continue

        universe = schema["universe"]
        pc_idx = schema["pc"] - 1
        features = schema["features"]

        prefixes = ALL_UNIVERSES[universe]
        indices = get_universe_indices(char_names, prefixes)
        u_names = [char_names[i] for i in indices]
        u_residuals = residuals[indices]

        # Refit PCA to get scores for selecting top/bottom chars
        u_pca = SkPCA(n_components=N_PCS)
        u_scores = u_pca.fit_transform(u_residuals)
        scores = u_scores[:, pc_idx]
        pc_dir = u_pca.components_[pc_idx]
        sorted_idx = np.argsort(scores)

        # Even-ranked characters for coding (2nd, 4th, ..., 20th from each end)
        high_idx, low_idx = strided_select(sorted_idx, N_CODE_PER_SIDE, stride_offset=1)
        chars_to_code = [u_names[i] for i in high_idx] + [u_names[i] for i in low_idx]

        # Get top 50 discriminative questions, sample 10 per character
        top_q_indices, _ = get_discriminative_questions(
            u_names, pc_dir, scaler, role_pca
        )
        disc_pool = top_q_indices[:N_DISCRIMINATIVE_POOL]

        if schema_key not in coded:
            coded[schema_key] = {"schema": schema, "characters": {}}

        n_done = sum(1 for c in chars_to_code if c in coded[schema_key]["characters"])
        n_total = len(chars_to_code)
        print(
            f"\n{schema_key}: {n_done}/{n_total} coded (even-ranked, top/bottom {N_CODE_PER_SIDE})"
        )

        for char_name in chars_to_code:
            pretty = char_name.split("__")[-1].replace("_", " ").title()
            if char_name in coded[schema_key]["characters"]:
                continue

            # Sample N_CODE_QUESTIONS from the discriminative pool
            q_sample = rng.choice(
                disc_pool, size=min(N_CODE_QUESTIONS, len(disc_pool)), replace=False
            ).tolist()

            prompt = build_code_prompt(char_name, q_sample, questions, features)
            if not prompt:
                coded[schema_key]["characters"][char_name] = {"error": "no responses"}
                continue

            try:
                response = call_llm(prompt, model=CODE_MODEL)
                json_str = response
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                ratings = json.loads(json_str.strip())
                coded[schema_key]["characters"][char_name] = ratings
                print(f"  {pretty}: {ratings}")

            except Exception as e:
                coded[schema_key]["characters"][char_name] = {"error": str(e)}
                print(f"  {pretty}: error - {e}")

            # Save periodically (every 10 characters)
            n_done = sum(
                1 for c in chars_to_code if c in coded[schema_key]["characters"]
            )
            if n_done % 10 == 0:
                with open(CODED_PATH, "w") as f:
                    json.dump(coded, f, indent=2, ensure_ascii=False)

            time.sleep(1)

        # Save after each universe-PC
        with open(CODED_PATH, "w") as f:
            json.dump(coded, f, indent=2, ensure_ascii=False)

    print(f"\nCoded data saved to {CODED_PATH}")
    return coded


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/analysis/llm_feature_coding.py [discover|code|all]")
        sys.exit(1)

    phase = sys.argv[1]
    char_data, questions, lu_data = load_data()

    if phase in ("discover", "all"):
        phase_discover(char_data, questions, lu_data)
    if phase in ("code", "all"):
        phase_code(char_data, questions, lu_data)


if __name__ == "__main__":
    main()
