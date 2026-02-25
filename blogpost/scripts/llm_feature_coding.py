"""LLM-based feature coding for PC interpretation.

Three modes:
  residual: PCA on residuals after projecting out role space (original behavior)
  within:   PCA on full centered activations within each universe
  lu:       Use role PCs, applied per-universe

Two-phase pipeline:
  Phase 1 (discover): For each universe × PC, show extreme characters' responses
    to an LLM and ask it to propose distinguishing features.
  Phase 2 (code): For each character in universe, show responses and ask LLM
    to rate on the discovered features.

Requires OPENROUTER_API_KEY environment variable.

Usage (from repo root):
    python blogpost/scripts/llm_feature_coding.py --mode residual discover
    python blogpost/scripts/llm_feature_coding.py --mode within all
    python blogpost/scripts/llm_feature_coding.py --mode lu discover
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
API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY_BATCH", os.environ.get("ANTHROPIC_API_KEY", "")
)
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_BATCH_URL = "https://api.anthropic.com/v1/messages/batches"
MODEL = "claude-sonnet-4-20250514"
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


def get_output_paths(mode, prompt_style="style"):
    suffix = f"_{mode}" if mode != "residual" else ""
    if prompt_style != "style":
        suffix += f"_{prompt_style}"
    return (
        f"results/llm_feature_schemas{suffix}.json",
        f"results/llm_feature_coded{suffix}.json",
    )


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


def call_llm(prompt, system="You are a careful linguistic analyst."):
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = {
        "model": MODEL,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.0,
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


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


def get_discriminative_questions(u_names, pc_dir, role_pca, mode="residual"):
    """Find questions with highest per-question variance along a PC direction.

    mode controls what activations the PC direction is projected onto:
      residual: project residuals (after removing Lu's space)
      within/lu: project full centered activations
    """
    all_projections = []
    for cname in u_names:
        acts = load_per_question_activations(cname)
        if acts is not None and len(acts) == 240:
            acts_centered = acts - role_pca.mean_
            if mode == "residual":
                in_role = acts_centered @ role_pca.components_.T
                acts_proj = acts_centered - in_role @ role_pca.components_
            else:
                acts_proj = acts_centered
            all_projections.append(acts_proj @ pc_dir)

    if not all_projections:
        return list(range(N_DISCOVER_QUESTIONS)), None

    all_proj = np.array(all_projections)
    question_variance = np.var(all_proj, axis=0)
    top_q_indices = np.argsort(question_variance)[::-1].tolist()
    return top_q_indices, question_variance


def get_universe_directions(mode, u_centered, residuals_u, role_pca, n_pcs):
    """Get PC directions and scores for a universe based on mode.

    Returns (scores, components) where:
      scores: (n_chars, n_pcs) array of PC scores
      components: (n_pcs, hidden_dim) array of PC directions
    """
    if mode == "residual":
        pca = SkPCA(n_components=n_pcs)
        scores = pca.fit_transform(residuals_u)
        return scores, pca.components_
    elif mode == "within":
        pca = SkPCA(n_components=n_pcs)
        scores = pca.fit_transform(u_centered)
        return scores, pca.components_
    elif mode == "lu":
        components = role_pca.components_[:n_pcs]
        scores = u_centered @ components.T
        return scores, components
    else:
        raise ValueError(f"Unknown mode: {mode}")


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

DISCOVER_PROMPT_STYLE = """Below are responses from two groups of anonymous characters to the same questions.
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

DISCOVER_PROMPT_OPEN = """Below are responses from two groups of anonymous characters to the same questions.
Group A contains 5 characters; Group B contains 5 characters. They are from the same fictional universe.

{response_pairs}

Based ONLY on the response text above (not any outside knowledge), propose exactly 6 features that
systematically distinguish Group A from Group B. Each feature should be:

1. Observable purely from the response text (not requiring knowledge of who the characters are)
2. A spectrum (not binary) -- something you could rate on a 1-5 scale

Features can be about communication style, topics discussed, attitudes expressed, values, personality,
or any other pattern you observe in the text.

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

DISCOVER_PROMPTS = {
    "style": DISCOVER_PROMPT_STYLE,
    "open": DISCOVER_PROMPT_OPEN,
}


def build_discovery_prompt(
    high_chars, low_chars, q_indices, questions, prompt_style="style"
):
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

    template = DISCOVER_PROMPTS[prompt_style]
    return template.format(response_pairs="\n".join(parts))


def phase_discover(
    char_data, questions, lu_data, mode="residual", prompt_style="style"
):
    print(f"=== Phase 1: Feature Discovery (mode={mode}, prompt={prompt_style}) ===\n")

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = lu_data["pca"]

    chars_centered = activation_matrix - role_pca.mean_
    chars_in_role_space = chars_centered @ role_pca.components_.T
    reconstructed = chars_in_role_space @ role_pca.components_
    residuals = chars_centered - reconstructed

    schema_path, _ = get_output_paths(mode, prompt_style)

    # Load existing schemas to allow resuming
    if Path(schema_path).exists():
        with open(schema_path) as f:
            schemas = json.load(f)
    else:
        schemas = {}

    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            continue

        u_centered = chars_centered[indices]
        u_residuals = residuals[indices]
        u_names = [char_names[i] for i in indices]

        u_scores, u_components = get_universe_directions(
            mode, u_centered, u_residuals, role_pca, N_PCS
        )

        for pc_idx in range(N_PCS):
            key = f"{universe}__PC{pc_idx + 1}"
            if key in schemas:
                print(f"Skipping {key} (already discovered)")
                continue

            scores = u_scores[:, pc_idx]
            pc_dir = u_components[pc_idx]
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
                u_names, pc_dir, role_pca, mode=mode
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

            prompt = build_discovery_prompt(
                high_chars, low_chars, usable_qs, questions, prompt_style
            )
            print(f"  Calling LLM for feature discovery ({len(prompt)} chars)...")

            try:
                response = call_llm(prompt)
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
            with open(schema_path, "w") as f:
                json.dump(schemas, f, indent=2, ensure_ascii=False)

            time.sleep(2)

    print(f"\nSchemas saved to {schema_path}")
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


def phase_code(char_data, questions, lu_data, mode="residual", prompt_style="style"):
    """Phase 2: submit all coding requests via Anthropic batch API, then poll."""
    print(
        f"=== Phase 2: Feature Coding via Batch API (mode={mode}, prompt={prompt_style}) ===\n"
    )

    schema_path, coded_path = get_output_paths(mode, prompt_style)

    if not Path(schema_path).exists():
        print(f"Error: {schema_path} not found. Run 'discover' first.")
        return None

    with open(schema_path) as f:
        schemas = json.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = lu_data["pca"]

    chars_centered = activation_matrix - role_pca.mean_
    chars_in_role_space = chars_centered @ role_pca.components_.T
    reconstructed = chars_in_role_space @ role_pca.components_
    residuals = chars_centered - reconstructed

    n_questions = len(questions)

    # Load existing coded data to allow resuming
    if Path(coded_path).exists():
        with open(coded_path) as f:
            coded = json.load(f)
    else:
        coded = {}

    rng = np.random.RandomState(42)

    # Build all requests
    batch_requests = []  # list of (custom_id, schema_key, char_name, request_body)

    for schema_key, schema in schemas.items():
        if "error" in schema:
            continue

        universe = schema["universe"]
        pc_idx = schema["pc"] - 1
        features = schema["features"]

        prefixes = ALL_UNIVERSES[universe]
        indices = get_universe_indices(char_names, prefixes)
        u_names = [char_names[i] for i in indices]
        u_centered = chars_centered[indices]
        u_residuals = residuals[indices]

        u_scores, u_components = get_universe_directions(
            mode, u_centered, u_residuals, role_pca, N_PCS
        )
        scores = u_scores[:, pc_idx]
        sorted_idx = np.argsort(scores)

        high_idx, low_idx = strided_select(sorted_idx, N_CODE_PER_SIDE, stride_offset=1)
        chars_to_code = [u_names[i] for i in high_idx] + [u_names[i] for i in low_idx]

        if schema_key not in coded:
            coded[schema_key] = {"schema": schema, "characters": {}}

        for char_name in chars_to_code:
            if char_name in coded[schema_key]["characters"]:
                continue

            # Random questions — subset sweep shows any 10 recover the signal
            q_sample = rng.choice(
                n_questions, size=N_CODE_QUESTIONS, replace=False
            ).tolist()

            prompt = build_code_prompt(char_name, q_sample, questions, features)
            if not prompt:
                coded[schema_key]["characters"][char_name] = {"error": "no responses"}
                continue

            req_idx = len(batch_requests)
            custom_id = f"req-{req_idx:04d}"
            batch_requests.append(
                (
                    custom_id,
                    schema_key,
                    char_name,
                    {
                        "custom_id": custom_id,
                        "params": {
                            "model": MODEL,
                            "max_tokens": 2000,
                            "temperature": 0.0,
                            "system": "You are a careful linguistic analyst.",
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    },
                )
            )

    if not batch_requests:
        print("All characters already coded.")
        return coded

    print(f"Submitting {len(batch_requests)} requests to Anthropic batch API...")

    # Write JSONL for batch
    batch_jsonl_path = coded_path.replace(".json", "_batch_input.jsonl")
    with open(batch_jsonl_path, "w") as f:
        for _, _, _, req in batch_requests:
            f.write(json.dumps(req) + "\n")

    # Submit batch
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Read the JSONL and submit
    with open(batch_jsonl_path, "r") as f:
        requests_list = [json.loads(line) for line in f]

    batch_resp = requests.post(
        ANTHROPIC_BATCH_URL,
        headers=headers,
        json={"requests": requests_list},
        timeout=120,
    )
    if batch_resp.status_code != 200:
        print(f"Batch API error {batch_resp.status_code}: {batch_resp.text[:1000]}")
        batch_resp.raise_for_status()
    batch_data = batch_resp.json()
    batch_id = batch_data["id"]
    print(f"Batch submitted: {batch_id}")

    # Poll for completion
    while True:
        time.sleep(30)
        status_resp = requests.get(
            f"{ANTHROPIC_BATCH_URL}/{batch_id}",
            headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
            timeout=30,
        )
        status_resp.raise_for_status()
        status = status_resp.json()
        counts = status.get("request_counts", {})
        processing = counts.get("processing", 0)
        succeeded = counts.get("succeeded", 0)
        errored = counts.get("errored", 0)
        print(
            f"  {status['processing_status']}: {succeeded} done, {processing} processing, {errored} errored"
        )

        if status["processing_status"] == "ended":
            break

    # Retrieve results
    results_url = status.get("results_url")
    if not results_url:
        print("Error: no results_url in batch response")
        return coded

    results_resp = requests.get(
        results_url,
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
        timeout=120,
    )
    results_resp.raise_for_status()

    # Build lookup from custom_id to (schema_key, char_name)
    id_to_info = {cid: (sk, cn) for cid, sk, cn, _ in batch_requests}

    # Parse JSONL results
    for line in results_resp.text.strip().split("\n"):
        result = json.loads(line)
        custom_id = result["custom_id"]
        schema_key, char_name = id_to_info[custom_id]
        pretty = char_name.split("__")[-1].replace("_", " ").title()

        if result["result"]["type"] == "succeeded":
            text = result["result"]["message"]["content"][0]["text"]
            json_str = text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            try:
                ratings = json.loads(json_str.strip())
                coded[schema_key]["characters"][char_name] = ratings
                print(f"  {pretty}: {ratings}")
            except json.JSONDecodeError as e:
                coded[schema_key]["characters"][char_name] = {"error": f"parse: {e}"}
                print(f"  {pretty}: parse error")
        else:
            error = result["result"].get("error", {}).get("message", "unknown")
            coded[schema_key]["characters"][char_name] = {"error": error}
            print(f"  {pretty}: API error - {error}")

    with open(coded_path, "w") as f:
        json.dump(coded, f, indent=2, ensure_ascii=False)

    print(f"\nCoded data saved to {coded_path}")
    return coded


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["discover", "code", "all"])
    parser.add_argument(
        "--mode",
        choices=["residual", "within", "lu"],
        default="residual",
        help="Which PC directions to interpret",
    )
    parser.add_argument(
        "--prompt",
        choices=["style", "open"],
        default="style",
        help="Discovery prompt: style (HOW only) or open (style + content)",
    )
    args = parser.parse_args()

    char_data, questions, lu_data = load_data()

    if args.phase in ("discover", "all"):
        phase_discover(
            char_data, questions, lu_data, mode=args.mode, prompt_style=args.prompt
        )
    if args.phase in ("code", "all"):
        phase_code(
            char_data, questions, lu_data, mode=args.mode, prompt_style=args.prompt
        )


if __name__ == "__main__":
    main()
