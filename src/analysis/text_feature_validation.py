"""
Non-LLM text feature validation for residual PC interpretations.

Computes simple text statistics from response texts and regresses them
against residual PC scores. This validates that the residual PCs capture
text-observable structure without relying on LLM-as-judge.

Also computes PCA on subsets of questions (top-10 most informative vs
random 10 vs all 240) to test whether a small number of informative
questions reproduce the same PCs.

Output: results/text_feature_validation.json
"""

import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler

BASE = Path("/workspace-vast/lnajt/persona_vectors/fictional-character-vectors")
RESPONSES_DIR = BASE / "outputs/qwen3-32b_20260211_002840/responses"
ACTIVATIONS_DIR = BASE / "outputs/qwen3-32b_20260211_002840/activations"
RESULTS_DIR = BASE / "results"

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


def load_responses(char_name):
    """Load all 240 responses for a character."""
    path = RESPONSES_DIR / f"{char_name}.jsonl"
    if not path.exists():
        return None
    responses = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            q_idx = entry["question_index"]
            text = entry["conversation"][-1]["content"]
            responses[q_idx] = text
    return responses


def compute_text_features(responses):
    """Compute non-LLM text features from response texts."""
    if not responses:
        return None

    all_texts = [responses[q] for q in sorted(responses)]

    # Per-response features, then average
    features = {}

    # 1. Response length (words)
    lengths = [len(t.split()) for t in all_texts]
    features["response_length"] = np.mean(lengths)

    # 2. First-person pronoun rate
    fp_pattern = re.compile(
        r"\b(i|me|my|mine|myself|i\'m|i\'ve|i\'d|i\'ll)\b", re.IGNORECASE
    )
    fp_counts = [len(fp_pattern.findall(t)) for t in all_texts]
    features["first_person_rate"] = np.mean(
        [c / max(len(t.split()), 1) for c, t in zip(fp_counts, all_texts)]
    )

    # 3. Third-person / impersonal pronoun rate
    tp_pattern = re.compile(
        r"\b(one|one\'s|they|them|their|those who|mortals?)\b", re.IGNORECASE
    )
    tp_counts = [len(tp_pattern.findall(t)) for t in all_texts]
    features["impersonal_rate"] = np.mean(
        [c / max(len(t.split()), 1) for c, t in zip(tp_counts, all_texts)]
    )

    # 4. Question marks (engagement / asking follow-ups)
    qmark_counts = [t.count("?") for t in all_texts]
    features["question_rate"] = np.mean(qmark_counts)

    # 5. Exclamation marks (expressiveness)
    exclam_counts = [t.count("!") for t in all_texts]
    features["exclamation_rate"] = np.mean(exclam_counts)

    # 6. Average sentence length (proxy for complexity)
    sent_lengths = []
    for t in all_texts:
        sents = re.split(r"[.!?]+", t)
        sents = [s.strip() for s in sents if s.strip()]
        if sents:
            sent_lengths.append(np.mean([len(s.split()) for s in sents]))
    features["avg_sentence_length"] = np.mean(sent_lengths) if sent_lengths else 0

    # 7. Casual markers
    casual_markers = [
        "yeah",
        "gonna",
        "gotta",
        "kinda",
        "sorta",
        "wanna",
        "hey",
        "lol",
        "haha",
        "like,",
        "you know",
        "right?",
        "i mean",
        "well,",
        "um",
        "uh",
        "ok ",
        "okay",
        "y'know",
        "ain't",
        "dunno",
        "lemme",
        "c'mon",
    ]
    casual_counts = [sum(t.lower().count(m) for m in casual_markers) for t in all_texts]
    features["casual_rate"] = np.mean(
        [c / max(len(t.split()), 1) for c, t in zip(casual_counts, all_texts)]
    )

    # 8. Formal/archaic markers
    formal_markers = [
        "thus",
        "therefore",
        "indeed",
        "furthermore",
        "moreover",
        "shall",
        "henceforth",
        "whereby",
        "therein",
        "behold",
        "verily",
        "whence",
        "doth",
        "thou",
        "thy",
        "thee",
        "hath",
        "whilst",
        "amongst",
        "ere",
    ]
    formal_counts = [sum(t.lower().count(m) for m in formal_markers) for t in all_texts]
    features["formal_rate"] = np.mean(
        [c / max(len(t.split()), 1) for c, t in zip(formal_counts, all_texts)]
    )

    # 9. Philosophical/abstract vocabulary
    phil_markers = [
        "existence",
        "truth",
        "essence",
        "cosmic",
        "eternal",
        "mortal",
        "destiny",
        "fate",
        "transcend",
        "divine",
        "infinite",
        "universe",
        "wisdom",
        "power",
        "reality",
        "void",
        "chaos",
        "order",
        "purpose",
        "meaning",
    ]
    phil_counts = [sum(t.lower().count(m) for m in phil_markers) for t in all_texts]
    features["philosophical_rate"] = np.mean(
        [c / max(len(t.split()), 1) for c, t in zip(phil_counts, all_texts)]
    )

    # 10. Concrete/practical vocabulary
    concrete_markers = [
        "actually",
        "specifically",
        "example",
        "step",
        "first",
        "then",
        "next",
        "try",
        "make sure",
        "basically",
        "practice",
        "simple",
        "easy",
        "just",
        "probably",
        "usually",
        "sometimes",
    ]
    concrete_counts = [
        sum(t.lower().count(m) for m in concrete_markers) for t in all_texts
    ]
    features["concrete_rate"] = np.mean(
        [c / max(len(t.split()), 1) for c, t in zip(concrete_counts, all_texts)]
    )

    # 11. Unique word ratio (vocabulary diversity)
    all_words = " ".join(all_texts).lower().split()
    features["unique_word_ratio"] = len(set(all_words)) / max(len(all_words), 1)

    # 12. Asterisk actions (*sighs*, *leans forward*)
    action_pattern = re.compile(r"\*[^*]+\*")
    action_counts = [len(action_pattern.findall(t)) for t in all_texts]
    features["action_rate"] = np.mean(action_counts)

    return features


def get_universe_indices(char_names, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def run_text_feature_regression():
    """Regress residual PC scores against non-LLM text features."""
    print("=== Non-LLM Text Feature Validation ===\n")

    # Load character data
    with open(RESULTS_DIR / "fictional_character_analysis_filtered.pkl", "rb") as f:
        char_data = pickle.load(f)
    with open(str(BASE / "data/role_vectors/qwen-3-32b_pca_layer32.pkl"), "rb") as f:
        role_data = pickle.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = role_data["pca"]
    role_scaler = role_data["scaler"]

    chars_scaled = role_scaler.transform(activation_matrix)
    chars_in_role_space = chars_scaled @ role_pca.components_.T @ role_pca.components_
    residuals = chars_scaled - chars_in_role_space

    # Load text features for all characters
    print("Loading response texts and computing features...")
    char_features = {}
    for i, name in enumerate(char_names):
        responses = load_responses(name)
        if responses:
            features = compute_text_features(responses)
            if features:
                char_features[name] = features
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(char_names)} characters processed")

    print(f"Computed features for {len(char_features)}/{len(char_names)} characters\n")

    feature_names = list(next(iter(char_features.values())).keys())

    results = {}
    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            continue

        u_residuals = residuals[indices]
        u_names = [char_names[i] for i in indices]

        # Residual PCA
        u_pca = SkPCA(n_components=2)
        u_transformed = u_pca.fit_transform(u_residuals)

        for pc in range(2):
            pc_scores = u_transformed[:, pc]

            # Build feature matrix for characters that have text features
            X_rows = []
            y_rows = []
            names_used = []
            for j, name in enumerate(u_names):
                if name in char_features:
                    X_rows.append([char_features[name][f] for f in feature_names])
                    y_rows.append(pc_scores[j])
                    names_used.append(name)

            if len(X_rows) < 10:
                continue

            X = np.array(X_rows)
            y = np.array(y_rows)

            # Standardize features
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)

            # Per-feature correlations
            correlations = []
            for f_idx, fname in enumerate(feature_names):
                r = np.corrcoef(X_std[:, f_idx], y)[0, 1]
                correlations.append({"feature": fname, "correlation": float(r)})
            correlations.sort(key=lambda x: -abs(x["correlation"]))

            # Multiple regression (OLS)
            X_with_intercept = np.column_stack([np.ones(len(X_std)), X_std])
            try:
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                y_hat = X_with_intercept @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - ss_res / ss_tot
                n, p = X_std.shape
                adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            except Exception:
                r_squared = None
                adj_r_squared = None

            key = f"{universe}__PC{pc + 1}"
            results[key] = {
                "universe": universe,
                "pc": pc + 1,
                "n_chars": len(X_rows),
                "r_squared": float(r_squared) if r_squared is not None else None,
                "adj_r_squared": float(adj_r_squared)
                if adj_r_squared is not None
                else None,
                "correlations": correlations,
                "feature_names": feature_names,
            }

            print(
                f"{universe} PC{pc + 1}: adj R² = {adj_r_squared:.3f} (n={len(X_rows)})"
            )
            top3 = ", ".join(
                c["feature"] + "(" + f"{c['correlation']:+.2f}" + ")"
                for c in correlations[:3]
            )
            print(f"  Top 3: {top3}")

    return results


def run_question_subset_pca():
    """Compare PCA from top-10 questions vs all 240 questions."""
    print("\n=== Question Subset PCA Comparison ===\n")

    import torch

    with open(RESULTS_DIR / "fictional_character_analysis_filtered.pkl", "rb") as f:
        char_data = pickle.load(f)
    with open(str(BASE / "data/role_vectors/qwen-3-32b_pca_layer32.pkl"), "rb") as f:
        role_data = pickle.load(f)
    with open(RESULTS_DIR / "blogpost_precomputed.json") as f:
        precomputed = json.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = role_data["pca"]
    role_scaler = role_data["scaler"]

    chars_scaled = role_scaler.transform(activation_matrix)

    results = {}

    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            continue

        u_names = [char_names[i] for i in indices]

        # Get top question indices from precomputed question_differentiation
        qd_data = precomputed["question_differentiation"].get(universe)
        if not qd_data or "top_questions" not in qd_data:
            print(f"  {universe}: no question differentiation data, skipping")
            continue

        # top_questions is [[q_index, variance, text], ...]
        top_q_list = qd_data["top_questions"]
        top10_indices = [int(q[0]) for q in top_q_list[:10]]
        # Use indices NOT in top questions as "bottom" proxy
        all_top_indices = set(int(q[0]) for q in top_q_list)
        remaining_indices = [i for i in range(240) if i not in all_top_indices]
        bottom10_indices = remaining_indices[-10:]  # least informative among remaining

        # Load per-question activations for each character
        print(
            f"Loading per-question activations for {universe} ({len(u_names)} characters)..."
        )
        per_q_acts = {}  # char_name -> {q_idx: activation_vector}

        for name in u_names:
            act_path = ACTIVATIONS_DIR / f"{name}.pt"
            if not act_path.exists():
                continue
            data = torch.load(act_path, map_location="cpu", weights_only=True)

            q_acts = {}
            for q_idx in range(240):
                positions = []
                for p_idx in range(5):
                    key = f"pos_p{p_idx}_q{q_idx}"
                    if key in data:
                        act = data[key].float()  # (64, 5120)
                        positions.append(act[31])  # layer 32 = index 31
                if positions:
                    q_acts[q_idx] = torch.stack(positions).mean(dim=0).numpy()
            per_q_acts[name] = q_acts

        if len(per_q_acts) < 20:
            print(
                f"  {universe}: only {len(per_q_acts)} characters with activations, skipping"
            )
            continue

        # Compute mean activation vectors for different question subsets
        names_with_acts = [n for n in u_names if n in per_q_acts]

        def compute_mean_vector(char_name, q_indices):
            acts = per_q_acts[char_name]
            vecs = [acts[q] for q in q_indices if q in acts]
            if not vecs:
                return None
            return np.mean(vecs, axis=0)

        def pca_from_subset(q_indices, label):
            vectors = []
            valid_names = []
            for name in names_with_acts:
                v = compute_mean_vector(name, q_indices)
                if v is not None:
                    vectors.append(v)
                    valid_names.append(name)
            if len(vectors) < 10:
                return None, None
            X = np.array(vectors)
            X_scaled = role_scaler.transform(X)
            pca = SkPCA(n_components=1)
            scores = pca.fit_transform(X_scaled).flatten()
            return scores, valid_names

        # All 240 questions
        all_scores, all_names = pca_from_subset(list(range(240)), "all 240")
        # Top 10 most informative
        top10_scores, _ = pca_from_subset(top10_indices, "top 10")
        # Bottom 10 least informative
        bot10_scores, _ = pca_from_subset(bottom10_indices, "bottom 10")
        # Random 10
        np.random.seed(42)
        rand10_indices = np.random.choice(240, 10, replace=False).tolist()
        rand10_scores, _ = pca_from_subset(rand10_indices, "random 10")

        if all_scores is None:
            continue

        # Ranking correlations
        corr_top10 = (
            abs(np.corrcoef(all_scores, top10_scores)[0, 1])
            if top10_scores is not None
            else None
        )
        corr_bot10 = (
            abs(np.corrcoef(all_scores, bot10_scores)[0, 1])
            if bot10_scores is not None
            else None
        )
        corr_rand10 = (
            abs(np.corrcoef(all_scores, rand10_scores)[0, 1])
            if rand10_scores is not None
            else None
        )

        results[universe] = {
            "n_chars": len(all_names),
            "top10_corr": float(corr_top10) if corr_top10 is not None else None,
            "bottom10_corr": float(corr_bot10) if corr_bot10 is not None else None,
            "random10_corr": float(corr_rand10) if corr_rand10 is not None else None,
            "top10_q_indices": top10_indices,
        }

        print(
            f"  {universe} (n={len(all_names)}): top10={corr_top10:.3f}, bot10={corr_bot10:.3f}, rand10={corr_rand10:.3f}"
        )

    return results


if __name__ == "__main__":
    # Part 1: Text feature regression
    text_results = run_text_feature_regression()

    # Save results
    output = {
        "text_feature_regression": text_results,
    }

    output_path = RESULTS_DIR / "text_feature_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
