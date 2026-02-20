"""Precompute all cluster-dependent data for blogpost.org.

Runs on the cluster (needs access to per-question activations and responses).
Saves everything to a single JSON file that the org file loads locally.

Usage:
    python src/analysis/precompute_blogpost_data.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA as SkPCA

# --- Paths ---
ACTIVATIONS_DIR = Path("outputs/qwen3-32b_20260211_002840/activations")
RESPONSES_DIR = Path("outputs/qwen3-32b_20260211_002840/responses")
CHAR_DATA_PATH = "results/fictional_character_analysis_filtered.pkl"
LU_PCA_PATH = "data/role_vectors/qwen-3-32b_pca_layer32.pkl"
QUESTIONS_PATH = "assistant-axis/data/extraction_questions.jsonl"
OUTPUT_PATH = "results/blogpost_precomputed.json"

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

N_EXTREME = 5

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
FIRST_PERSON = ["i ", "i'm", "i've", "i'll", "i'd", "my ", "me ", "myself"]
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


def count_markers(text, markers):
    text_lower = text.lower()
    return sum(text_lower.count(m) for m in markers)


def extract_features(responses_dict):
    all_texts = [t for qi in sorted(responses_dict) for t in responses_dict[qi]]
    if not all_texts:
        return None
    return {
        "casual": float(np.mean([count_markers(t, CASUAL_MARKERS) for t in all_texts])),
        "first_person": float(
            np.mean([count_markers(t, FIRST_PERSON) for t in all_texts])
        ),
        "philosophical": float(
            np.mean([count_markers(t, PHILOSOPHICAL) for t in all_texts])
        ),
        "exclamation": float(np.mean(["!" in t for t in all_texts])),
    }


def compute_question_differentiation(char_names, residuals, questions):
    """Block 1: For each universe, find top differentiating questions between high/low on residual PC1."""
    print("Computing question differentiation...")
    results = {}

    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            continue

        u_residuals = residuals[indices]
        u_names = [char_names[i] for i in indices]

        u_pca = SkPCA(n_components=1)
        scores = u_pca.fit_transform(u_residuals)[:, 0]
        sorted_idx = np.argsort(scores)

        high_names = [u_names[i] for i in sorted_idx[-5:]]
        low_names = [u_names[i] for i in sorted_idx[:5]]

        high_labels = [n.split("__")[-1].replace("_", " ").title() for n in high_names]
        low_labels = [n.split("__")[-1].replace("_", " ").title() for n in low_names]

        # Load per-question activations for high/low chars
        high_acts = {n: load_per_question_activations(n) for n in high_names}
        low_acts = {n: load_per_question_activations(n) for n in low_names}

        diffs = []
        for q_idx in range(240):
            h_vecs = [
                high_acts[n][q_idx] for n in high_acts if high_acts[n] is not None
            ]
            l_vecs = [low_acts[n][q_idx] for n in low_acts if low_acts[n] is not None]
            if h_vecs and l_vecs:
                diff_norm = float(
                    np.linalg.norm(np.mean(h_vecs, axis=0) - np.mean(l_vecs, axis=0))
                )
                diffs.append((q_idx, diff_norm))
        diffs.sort(key=lambda x: x[1], reverse=True)
        top_qs = [(q_idx, diff, questions[q_idx]) for q_idx, diff in diffs[:5]]

        results[universe] = {
            "high_labels": high_labels,
            "low_labels": low_labels,
            "top_questions": top_qs,
        }
        print(f"  {universe}: done")

    return results


def compute_question_attribution(char_names, residuals, scaler, role_pca, questions):
    """Blocks 2-4: Per-question variance decomposition, cumvar curves, per-character attributions."""
    print("Computing question attribution (this is the slow part)...")
    results = {}

    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            continue

        u_residuals = residuals[indices]
        u_names = [char_names[i] for i in indices]

        u_pca = SkPCA(n_components=1)
        scores = u_pca.fit_transform(u_residuals)[:, 0]
        pc1_dir = u_pca.components_[0]
        sorted_idx = np.argsort(scores)

        # Load per-question activations for all chars in universe
        all_projections = []
        valid_names = []
        for cname in u_names:
            acts = load_per_question_activations(cname)
            if acts is not None and len(acts) == 240:
                acts_scaled = scaler.transform(acts)
                in_role = role_pca.transform(acts_scaled)
                acts_resid = acts_scaled - in_role @ role_pca.components_
                projections = (acts_resid @ pc1_dir).tolist()
                all_projections.append(projections)
                valid_names.append(cname)

        all_projections_arr = np.array(all_projections)
        question_variance = np.var(all_projections_arr, axis=0)

        # Cumulative variance curve
        sorted_var = np.sort(question_variance)[::-1]
        cumvar = (np.cumsum(sorted_var) / sorted_var.sum()).tolist()

        n50 = int(np.searchsorted(cumvar, 0.5) + 1)
        n80 = int(np.searchsorted(cumvar, 0.8) + 1)
        n90 = int(np.searchsorted(cumvar, 0.9) + 1)

        top_q_idx = np.argsort(question_variance)[::-1][:10].tolist()
        top1_pct = float(
            question_variance[top_q_idx[0]] / question_variance.sum() * 100
        )

        high_labels = [
            u_names[i].split("__")[-1].replace("_", " ").title()
            for i in sorted_idx[-5:]
        ]
        low_labels = [
            u_names[i].split("__")[-1].replace("_", " ").title() for i in sorted_idx[:5]
        ]

        # Per-character attributions (highest and lowest scoring among valid chars)
        sorted_chars = sorted(
            range(len(valid_names)), key=lambda i: np.mean(all_projections[i])
        )
        char_attributions = {}
        for char_idx, label in [(sorted_chars[-1], "HIGH"), (sorted_chars[0], "LOW")]:
            cname = valid_names[char_idx]
            pretty_name = cname.split("__")[-1].replace("_", " ").title()
            char_projs = np.array(all_projections[char_idx])
            sorted_q = np.argsort(char_projs)

            pushes_high = []
            for qi in sorted_q[-5:][::-1]:
                pushes_high.append(
                    {
                        "proj": float(char_projs[qi]),
                        "question": questions[qi][:70],
                    }
                )
            pushes_low = []
            for qi in sorted_q[:5]:
                pushes_low.append(
                    {
                        "proj": float(char_projs[qi]),
                        "question": questions[qi][:70],
                    }
                )

            char_attributions[label] = {
                "name": pretty_name,
                "mean_proj": float(char_projs.mean()),
                "pushes_high": pushes_high,
                "pushes_low": pushes_low,
            }

        results[universe] = {
            "cumvar": cumvar,
            "n50": n50,
            "n80": n80,
            "n90": n90,
            "top1_pct": top1_pct,
            "high_labels": high_labels,
            "low_labels": low_labels,
            "char_attributions": char_attributions,
        }
        print(f"  {universe}: done (n50={n50}, n80={n80}, n90={n90})")

    return results


def compute_sample_responses(char_names, questions):
    """Block 5: Load sample responses for hardcoded character/question pairs."""
    print("Computing sample responses...")

    samples = [
        (
            "Harry Potter",
            "Describe what happiness means to you.",
            ["harry_potter__ron_weasley", "harry_potter__lord_voldemort"],
        ),
        (
            "Star Wars",
            "How should we approach questions about the nature of existence?",
            ["star_wars__yoda", "star_wars__han_solo"],
        ),
        (
            "LOTR",
            "How do you feel about attending weddings or large family gatherings?",
            ["lord_of_the_rings__galadriel", "lord_of_the_rings__bilbo_baggins"],
        ),
        (
            "Marvel",
            "Can you explain why we need to drink water?",
            ["marvel__spiderman_peter_parker", "marvel__thanos"],
        ),
        (
            "Game of Thrones",
            "Your coworker's keyboard typing seems louder than usual today. What do you think about this?",
            [
                "game_of_thrones__tormund_giantsbane",
                "game_of_thrones__viserys_targaryen",
            ],
        ),
        (
            "Naruto",
            "What is the concept of happiness?",
            ["naruto__naruto_uzumaki", "naruto__madara_uchiha"],
        ),
        (
            "Greek Mythology",
            "If you could preserve one moment forever, what would it be?",
            ["greek_mythology__agamemnon", "greek_mythology__chaos"],
        ),
        (
            "Chinese Mythology",
            "How would you describe the perfect way to spend a weekend?",
            ["chinese_mythology__zhu_bajie", "chinese_mythology__phoenix"],
        ),
        (
            "Hindu Mythology",
            "If you could preserve one moment forever, what would it be?",
            ["hindu_mythology__arjuna", "hindu_mythology__shiva"],
        ),
        (
            "Norse Mythology",
            "How would you describe the perfect way to spend a weekend?",
            ["norse_mythology__erik_the_red", "norse_mythology__urd"],
        ),
        (
            "Egyptian Mythology",
            "How would you describe what makes life meaningful?",
            ["egyptian_mythology__tutankhamun", "egyptian_mythology__apep"],
        ),
        (
            "Shakespeare",
            "How do you understand the concept of meaning in life?",
            ["shakespeare__julius_caesar", "shakespeare__dogberry"],
        ),
    ]

    results = []
    for universe, q_text, chars in samples:
        qi = questions.index(q_text)
        sample = {"universe": universe, "question": q_text, "responses": []}
        for char in chars:
            name = char.split("__")[-1].replace("_", " ").title()
            resp = load_responses(char)
            text = None
            if resp and qi in resp:
                text = resp[qi][0]
            sample["responses"].append({"name": name, "char_id": char, "text": text})
        results.append(sample)
        print(f"  {universe}: done")

    return results


def compute_text_features(char_names, residuals):
    """Block 6: Aggregate text features (casual, first_person, etc.) for HIGH vs LOW chars on PC1 and PC2."""
    print("Computing aggregate text features...")
    results = []

    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            continue

        u_residuals = residuals[indices]
        u_names = [char_names[i] for i in indices]

        u_pca = SkPCA(n_components=2)
        u_scores = u_pca.fit_transform(u_residuals)

        for pc_idx in range(2):
            scores = u_scores[:, pc_idx]
            sorted_idx = np.argsort(scores)

            high_chars = [u_names[i] for i in sorted_idx[-N_EXTREME:][::-1]]
            low_chars = [u_names[i] for i in sorted_idx[:N_EXTREME]]

            high_feats, low_feats = [], []
            for name in high_chars:
                resp = load_responses(name)
                if resp:
                    f = extract_features(resp)
                    if f:
                        high_feats.append(f)
            for name in low_chars:
                resp = load_responses(name)
                if resp:
                    f = extract_features(resp)
                    if f:
                        low_feats.append(f)

            if not high_feats or not low_feats:
                continue

            pretty = lambda names: ", ".join(
                n.split("__")[-1].replace("_", " ").title() for n in names[:3]
            )
            row = {
                "universe": universe,
                "pc": pc_idx + 1,
                "high_end": pretty(high_chars),
                "low_end": pretty(low_chars),
            }
            for key in ["casual", "first_person", "philosophical", "exclamation"]:
                h = float(np.mean([f[key] for f in high_feats]))
                l = float(np.mean([f[key] for f in low_feats]))
                row[key] = float(h - l)
            results.append(row)
            print(f"  {universe} PC{pc_idx + 1}: done")

    return results


def compute_qualitative_examples(char_names, questions):
    """Block 7: Paired responses for qualitative comparison."""
    print("Computing qualitative examples...")
    q_text = "How do you stay organized in daily life?"
    qi = questions.index(q_text)

    pairs = [
        ("Harry Potter", "harry_potter__ron_weasley", "harry_potter__tom_riddle"),
        ("Naruto", "naruto__naruto_uzumaki", "naruto__madara_uchiha"),
        (
            "Game of Thrones",
            "game_of_thrones__tormund_giantsbane",
            "game_of_thrones__mad_king_aerys_targaryen",
        ),
    ]

    results = {"question": q_text, "pairs": []}
    for universe, high_char, low_char in pairs:
        pair_data = {"universe": universe, "responses": []}
        for label, char in [("HIGH", high_char), ("LOW", low_char)]:
            name = char.split("__")[-1].replace("_", " ").title()
            resp = load_responses(char)
            text = None
            if resp and qi in resp:
                text = resp[qi][0]
                words = text.split()
                if len(words) > 200:
                    text = " ".join(words[:200]) + " [...]"
            pair_data["responses"].append(
                {
                    "label": label,
                    "name": name,
                    "char_id": char,
                    "text": text,
                }
            )
        results["pairs"].append(pair_data)
        print(f"  {universe}: done")

    return results


def compute_llm_prompt_demo(char_names, residuals, scaler, role_pca, questions):
    """Block 8: LLM prompt demo data (shows prompt structure for Harry Potter)."""
    print("Computing LLM prompt demo...")

    N_QUESTIONS = 100

    universe = "Harry Potter"
    prefixes = ALL_UNIVERSES[universe]
    indices = get_universe_indices(char_names, prefixes)
    u_residuals = residuals[indices]
    u_names = [char_names[i] for i in indices]

    u_pca = SkPCA(n_components=1)
    scores = u_pca.fit_transform(u_residuals)[:, 0]
    pc1_dir = u_pca.components_[0]
    sorted_idx = np.argsort(scores)

    high_chars = [u_names[i] for i in sorted_idx[-N_EXTREME:][::-1]]
    low_chars = [u_names[i] for i in sorted_idx[:N_EXTREME]]

    pretty = lambda n: n.split("__")[-1].replace("_", " ").title()

    # Load per-question activations to find most discriminative questions
    all_projections = []
    for cname in u_names:
        acts = load_per_question_activations(cname)
        if acts is not None and len(acts) == 240:
            acts_scaled = scaler.transform(acts)
            in_role = role_pca.transform(acts_scaled)
            acts_resid = acts_scaled - in_role @ role_pca.components_
            all_projections.append(acts_resid @ pc1_dir)

    if all_projections:
        all_projections_arr = np.array(all_projections)
        question_variance = np.var(all_projections_arr, axis=0)
        top_q_indices = np.argsort(question_variance)[::-1][:N_QUESTIONS].tolist()
        variance_range = (
            float(question_variance[top_q_indices[0]]),
            float(question_variance[top_q_indices[-1]]),
        )
    else:
        top_q_indices = list(range(N_QUESTIONS))
        variance_range = None

    # Load responses for high/low chars
    high_resp = {}
    for n in high_chars:
        r = load_responses(n)
        if r:
            high_resp[n] = r
    low_resp = {}
    for n in low_chars:
        r = load_responses(n)
        if r:
            low_resp[n] = r

    usable_qs = [
        int(qi)
        for qi in top_q_indices
        if any(qi in r for r in high_resp.values())
        and any(qi in r for r in low_resp.values())
    ]

    # Get first 2 example Q&A pairs for the prompt demo
    example_pairs = []
    for i, qi in enumerate(usable_qs[:2]):
        q_text = questions[qi] if qi < len(questions) else f"question {qi}"
        pair = {"question_num": i + 1, "question": q_text}
        for name, resp in high_resp.items():
            if qi in resp:
                pair["group_a_response"] = resp[qi][0][:300] + "..."
                break
        for name, resp in low_resp.items():
            if qi in resp:
                pair["group_b_response"] = resp[qi][0][:300] + "..."
                break
        example_pairs.append(pair)

    return {
        "universe": universe,
        "group_a_high": [pretty(n) for n in high_chars],
        "group_b_low": [pretty(n) for n in low_chars],
        "top_5_q_indices": top_q_indices[:5],
        "variance_range": variance_range,
        "n_usable_qs": len(usable_qs),
        "example_pairs": example_pairs,
    }


def main():
    char_data, questions, lu_data = load_data()
    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = lu_data["pca"]
    scaler = lu_data["scaler"]

    # Compute residuals
    chars_scaled = scaler.transform(activation_matrix)
    chars_in_role_space = role_pca.transform(chars_scaled)
    reconstructed = chars_in_role_space @ role_pca.components_
    residuals = chars_scaled - reconstructed

    output = {}

    output["question_differentiation"] = compute_question_differentiation(
        char_names, residuals, questions
    )

    output["question_attribution"] = compute_question_attribution(
        char_names, residuals, scaler, role_pca, questions
    )

    output["sample_responses"] = compute_sample_responses(char_names, questions)

    output["text_features"] = compute_text_features(char_names, residuals)

    output["qualitative_examples"] = compute_qualitative_examples(char_names, questions)

    output["llm_prompt_demo"] = compute_llm_prompt_demo(
        char_names, residuals, scaler, role_pca, questions
    )

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
