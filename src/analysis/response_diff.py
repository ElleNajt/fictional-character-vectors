"""Systematic response diffing for residual PC interpretation.

For each universe's residual PC1:
  - Take top 3 and bottom 3 characters
  - Load their responses to the same questions
  - Compute aggregate text features (length, vocab, formality markers)
  - Sample representative response pairs for qualitative comparison

Outputs:
  results/response_diff.txt - quantitative features + sampled response pairs
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA

# --- Setup ---
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


def get_universe_indices(prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def load_responses(char_name):
    """Load all responses for a character. Returns dict: question_index -> list of response texts."""
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


# --- Text feature extraction ---

# Formality markers (casual vs formal)
CASUAL_MARKERS = [
    "lol",
    "haha",
    "yeah",
    "nah",
    "gonna",
    "wanna",
    "gotta",
    "kinda",
    "sorta",
    "dunno",
    "ya",
    "hey",
    "man",
    "dude",
    "bro",
    "stuff",
    "cool",
    "awesome",
    "ok",
    "okay",
    "umm",
    "hmm",
    "well,",
    "like,",
    "you know",
    "i mean",
    "pretty much",
    "no way",
    "for real",
]

FORMAL_MARKERS = [
    "furthermore",
    "moreover",
    "nevertheless",
    "consequently",
    "thus",
    "therefore",
    "henceforth",
    "wherein",
    "whereby",
    "thereof",
    "indeed",
    "certainly",
    "undoubtedly",
    "shall",
    "ought",
    "it is imperative",
    "one must",
    "it behooves",
    "in accordance",
    "with regard to",
    "in the matter of",
    "i would posit",
]

FIRST_PERSON = ["i ", "i'm", "i've", "i'll", "i'd", "my ", "me ", "myself"]
EMOTIONAL = [
    "feel",
    "heart",
    "soul",
    "love",
    "fear",
    "hope",
    "dream",
    "passion",
    "joy",
    "sorrow",
    "anger",
    "pain",
    "believe",
]
PHILOSOPHICAL = [
    "existence",
    "truth",
    "reality",
    "nature",
    "essence",
    "cosmic",
    "eternal",
    "infinite",
    "transcend",
    "universal",
    "fundamental",
    "primordial",
    "destiny",
    "fate",
]


def count_markers(text, markers):
    text_lower = text.lower()
    return sum(text_lower.count(m) for m in markers)


def extract_features(responses_dict):
    """Extract aggregate text features across all questions."""
    all_texts = []
    for qi in sorted(responses_dict.keys()):
        for text in responses_dict[qi]:
            all_texts.append(text)

    if not all_texts:
        return None

    lengths = [len(t.split()) for t in all_texts]
    features = {
        "n_responses": len(all_texts),
        "mean_length": np.mean(lengths),
        "median_length": np.median(lengths),
        "std_length": np.std(lengths),
        "casual_per_response": np.mean(
            [count_markers(t, CASUAL_MARKERS) for t in all_texts]
        ),
        "formal_per_response": np.mean(
            [count_markers(t, FORMAL_MARKERS) for t in all_texts]
        ),
        "first_person_per_response": np.mean(
            [count_markers(t, FIRST_PERSON) for t in all_texts]
        ),
        "emotional_per_response": np.mean(
            [count_markers(t, EMOTIONAL) for t in all_texts]
        ),
        "philosophical_per_response": np.mean(
            [count_markers(t, PHILOSOPHICAL) for t in all_texts]
        ),
        "exclamation_rate": np.mean(["!" in t for t in all_texts]),
        "question_rate": np.mean(["?" in t for t in all_texts]),
        "paragraph_count": np.mean([t.count("\n\n") + 1 for t in all_texts]),
    }
    return features


# --- Main ---
out = []
N_EXTREME = 3  # characters per end
N_SAMPLE_QUESTIONS = 5  # questions to show response pairs for

for universe, prefixes in ALL_UNIVERSES.items():
    indices = get_universe_indices(prefixes)
    if len(indices) < 20:
        continue

    u_residuals = residuals[indices]
    u_names = [char_names[i] for i in indices]

    u_pca = SkPCA(n_components=1)
    scores = u_pca.fit_transform(u_residuals)[:, 0]

    sorted_idx = np.argsort(scores)
    high_chars = [u_names[i] for i in sorted_idx[-N_EXTREME:][::-1]]
    low_chars = [u_names[i] for i in sorted_idx[:N_EXTREME]]
    high_scores = [scores[sorted_idx[-N_EXTREME + j]] for j in range(N_EXTREME)][::-1]
    low_scores = [scores[sorted_idx[j]] for j in range(N_EXTREME)]

    header = f"\n{'#' * 70}\n# {universe}\n{'#' * 70}"
    header += f"\n  PC1 explains {u_pca.explained_variance_ratio_[0]:.1%} of within-universe residual variance"
    out.append(header)
    print(header)

    # Display extreme characters
    out.append("\n  HIGH end:")
    for name, score in zip(high_chars, high_scores):
        pretty = name.split("__")[-1].replace("_", " ").title()
        out.append(f"    {score:+6.1f}  {pretty}")
    out.append("  LOW end:")
    for name, score in zip(low_chars, low_scores):
        pretty = name.split("__")[-1].replace("_", " ").title()
        out.append(f"    {score:+6.1f}  {pretty}")

    # Load responses and extract features
    high_features = []
    low_features = []
    high_responses = {}
    low_responses = {}

    for name in high_chars:
        resp = load_responses(name)
        if resp is None:
            print(f"  WARNING: no responses for {name}")
            continue
        high_responses[name] = resp
        feats = extract_features(resp)
        if feats:
            high_features.append(feats)

    for name in low_chars:
        resp = load_responses(name)
        if resp is None:
            print(f"  WARNING: no responses for {name}")
            continue
        low_responses[name] = resp
        feats = extract_features(resp)
        if feats:
            low_features.append(feats)

    if not high_features or not low_features:
        out.append("  (insufficient response data)")
        continue

    # Compare aggregate features
    out.append("\n  === Aggregate Text Features ===")
    out.append(f"  {'Feature':35s} {'HIGH mean':>12s} {'LOW mean':>12s} {'Diff':>10s}")
    out.append(f"  {'-' * 70}")

    for key in [
        "mean_length",
        "casual_per_response",
        "formal_per_response",
        "first_person_per_response",
        "emotional_per_response",
        "philosophical_per_response",
        "exclamation_rate",
        "question_rate",
        "paragraph_count",
    ]:
        h_val = np.mean([f[key] for f in high_features])
        l_val = np.mean([f[key] for f in low_features])
        diff = h_val - l_val
        line = f"  {key:35s} {h_val:12.2f} {l_val:12.2f} {diff:+10.2f}"
        out.append(line)
        print(line)

    # Sample response pairs for qualitative comparison
    # Pick 5 questions where the per-question features differ most
    out.append(f"\n  === Sample Response Pairs (first prompt only) ===")

    # Find questions where both groups have data
    common_qs = set()
    for resp in high_responses.values():
        common_qs.update(resp.keys())
    for resp in low_responses.values():
        common_qs.intersection_update(resp.keys())

    # For each common question, compute mean response length difference
    q_diffs = []
    for qi in sorted(common_qs):
        h_lens = []
        l_lens = []
        for resp in high_responses.values():
            if qi in resp:
                h_lens.extend(len(t.split()) for t in resp[qi])
        for resp in low_responses.values():
            if qi in resp:
                l_lens.extend(len(t.split()) for t in resp[qi])
        if h_lens and l_lens:
            q_diffs.append((qi, np.mean(h_lens) - np.mean(l_lens)))

    # Sort by absolute difference, take top N
    q_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    sample_qs = [qi for qi, _ in q_diffs[:N_SAMPLE_QUESTIONS]]

    for qi in sample_qs:
        q_text = questions[qi] if qi < len(questions) else f"question {qi}"
        out.append(f"\n  Q{qi}: {q_text}")
        out.append(f"  {'~' * 60}")

        # Show first HIGH character's first prompt response
        for name in high_chars:
            if name in high_responses and qi in high_responses[name]:
                pretty = name.split("__")[-1].replace("_", " ").title()
                text = high_responses[name][qi][0]  # first prompt variant
                # Truncate to 300 words
                words = text.split()
                if len(words) > 300:
                    text = " ".join(words[:300]) + " [...]"
                out.append(f"\n    HIGH [{pretty}] ({len(words)} words):")
                for line in text.split("\n"):
                    out.append(f"      {line}")
                break  # just first character

        for name in low_chars:
            if name in low_responses and qi in low_responses[name]:
                pretty = name.split("__")[-1].replace("_", " ").title()
                text = low_responses[name][qi][0]
                words = text.split()
                if len(words) > 300:
                    text = " ".join(words[:300]) + " [...]"
                out.append(f"\n    LOW [{pretty}] ({len(words)} words):")
                for line in text.split("\n"):
                    out.append(f"      {line}")
                break

Path("results/response_diff.txt").write_text("\n".join(out))
print(f"\nSaved results/response_diff.txt ({len(out)} lines)")
