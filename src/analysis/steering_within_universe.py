"""
Within-universe steering experiment: Do within-universe PC1 and residual PC1
produce different effects when steering characters from that universe?

For each universe with enough characters (>=20), computes:
  - Within-universe PC1: PCA on chars_scaled[universe_idx] (full space, includes Lu)
  - Residual PC1: PCA on residuals[universe_idx] (orthogonal to Lu's 275 dims)
  - Random direction (control)

For each universe, picks 3 test characters and steers with each direction type.
All directions scaled to Lu's axis norm (~22.7).

Usage:
    CUDA_VISIBLE_DEVICES=5,6 python3 src/analysis/steering_within_universe.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA as SkPCA

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "assistant-axis"))
from assistant_axis.internals.model import ProbingModel
from assistant_axis.steering import ActivationSteering

BASE = Path("/workspace-vast/lnajt/persona_vectors/fictional-character-vectors")
RESULTS_DIR = BASE / "results"

LAYER = 31  # 0-indexed

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

# Pick recognizable characters from each universe for readable results
UNIVERSE_TEST_CHARS = {
    "Harry Potter": [
        "harry_potter__harry_potter",
        "harry_potter__hermione_granger",
        "harry_potter__severus_snape",
    ],
    "Marvel": [
        "marvel__iron_man_tony_stark",
        "marvel__spiderman_peter_parker",
        "marvel__thanos",
    ],
    "LOTR": [
        "lord_of_the_rings__gandalf",
        "lord_of_the_rings__aragorn",
        "lord_of_the_rings__gollum_smeagol",
    ],
    "Game of Thrones": [
        "game_of_thrones__tyrion_lannister",
        "game_of_thrones__daenerys_targaryen",
        "game_of_thrones__jon_snow",
    ],
    "Greek Mythology": [
        "greek_mythology__zeus",
        "greek_mythology__athena",
        "greek_mythology__prometheus",
    ],
}

TEST_QUESTIONS = [
    "What is the relationship between law and morality?",
    "Describe what happiness means to you.",
]

COEFFICIENTS = [-10.0, -5.0, 5.0, 10.0]


def get_universe_indices(char_names, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def make_system_prompt(char_name):
    """Create system prompt from character name."""
    parts = char_name.split("__")
    universe = parts[0].replace("_", " ").title()
    character = parts[1].replace("_", " ").title()
    return f"You are {character} from {universe}. Respond as this character would."


def load_directions():
    """Compute within-universe PC1 and residual PC1 for each test universe."""
    # Load Lu's axis for norm reference
    axis_path = "/workspace-vast/lnajt/hf_cache/hub/datasets--lu-christina--assistant-axis-vectors/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b/assistant_axis.pt"
    lu_axis = torch.load(axis_path, map_location="cpu", weights_only=True)
    lu_axis_32 = lu_axis[32].float().numpy()
    lu_norm = np.linalg.norm(lu_axis_32)

    # Load character data
    with open(RESULTS_DIR / "fictional_character_analysis_filtered.pkl", "rb") as f:
        char_data = pickle.load(f)
    with open(str(BASE / "data/role_vectors/qwen-3-32b_pca_layer32.pkl"), "rb") as f:
        role_data = pickle.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = role_data["pca"]
    lu_scaler = role_data["scaler"]

    # Compute residuals
    chars_scaled = lu_scaler.transform(activation_matrix)
    chars_in_role_space = chars_scaled @ role_pca.components_.T @ role_pca.components_
    residuals = chars_scaled - chars_in_role_space

    print(f"Lu axis norm: {lu_norm:.1f}")
    print(f"Characters: {len(char_names)}, Dimensions: {chars_scaled.shape[1]}")
    print(f"Lu PCA components: {role_pca.n_components_}")
    print()

    universe_directions = {}

    for universe, prefixes in ALL_UNIVERSES.items():
        if universe not in UNIVERSE_TEST_CHARS:
            continue

        indices = get_universe_indices(char_names, prefixes)
        if len(indices) < 20:
            print(f"Skipping {universe}: only {len(indices)} characters")
            continue

        # Within-universe PC1 (full space, includes Lu's dimensions)
        u_scaled = chars_scaled[indices]
        u_pca = SkPCA(n_components=1)
        u_pca.fit(u_scaled)
        within_pc1 = u_pca.components_[0]
        within_pc1_unit = within_pc1 / np.linalg.norm(within_pc1)

        # Residual PC1 (orthogonal to Lu's 275 dims)
        u_residuals = residuals[indices]
        r_pca = SkPCA(n_components=1)
        r_pca.fit(u_residuals)
        resid_pc1 = r_pca.components_[0]
        resid_pc1_unit = resid_pc1 / np.linalg.norm(resid_pc1)

        # Random direction (control)
        rng = np.random.RandomState(hash(universe) % 2**31)
        random_dir = rng.randn(5120).astype(np.float32)
        random_dir_unit = random_dir / np.linalg.norm(random_dir)

        cosine = np.dot(within_pc1_unit, resid_pc1_unit)
        print(f"{universe} ({len(indices)} chars):")
        print(f"  Within PC1 var explained: {u_pca.explained_variance_ratio_[0]:.1%}")
        print(f"  Residual PC1 var explained: {r_pca.explained_variance_ratio_[0]:.1%}")
        print(f"  Cosine(within PC1, residual PC1): {cosine:.4f}")

        universe_directions[universe] = {
            "within_pc1": within_pc1_unit * lu_norm,
            "residual_pc1": resid_pc1_unit * lu_norm,
            "random": random_dir_unit * lu_norm,
        }

    return universe_directions, char_names


def generate_with_steering(pm, system_prompt, question, direction, coefficient, layer):
    """Generate a response with optional activation steering."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    formatted = pm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = pm.tokenizer(formatted, return_tensors="pt").to(pm.device)

    if coefficient == 0.0 or direction is None:
        with torch.no_grad():
            outputs = pm.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=pm.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
    else:
        direction_tensor = torch.tensor(direction, dtype=torch.bfloat16)
        with ActivationSteering(
            pm.model,
            steering_vectors=[direction_tensor],
            coefficients=[coefficient],
            layer_indices=[layer],
            intervention_type="addition",
            positions="all",
        ):
            with torch.no_grad():
                outputs = pm.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=pm.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

    response = pm.tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return response.strip()


def main():
    print("=== Within-Universe Steering Experiment ===")
    print("Comparing within-universe PC1 vs residual PC1 vs random\n")

    # Load directions
    universe_directions, char_names = load_directions()

    # Load model
    print("\nLoading Qwen3-32B...")
    pm = ProbingModel(
        "Qwen/Qwen3-32B",
        dtype=torch.bfloat16,
    )
    print(f"Model loaded. Hidden size: {pm.hidden_size}")

    results = []

    for universe, chars in UNIVERSE_TEST_CHARS.items():
        if universe not in universe_directions:
            print(f"\nSkipping {universe}: no directions computed")
            continue

        dirs = universe_directions[universe]
        print(f"\n{'=' * 60}")
        print(f"Universe: {universe}")
        print(f"{'=' * 60}")

        conditions = [
            ("baseline", None, [0.0]),
            ("within_pc1", dirs["within_pc1"], COEFFICIENTS),
            ("residual_pc1", dirs["residual_pc1"], COEFFICIENTS),
            ("random", dirs["random"], COEFFICIENTS),
        ]

        for char_name in chars:
            if char_name not in char_names:
                print(f"  WARNING: {char_name} not found in data, skipping")
                continue

            system_prompt = make_system_prompt(char_name)
            print(f"\n  --- {char_name} ---")

            for question in TEST_QUESTIONS:
                print(f"  Q: {question[:50]}...")

                for cond_name, direction, coeffs in conditions:
                    for coeff in coeffs:
                        if cond_name == "baseline" and coeff != 0.0:
                            continue

                        label = (
                            f"{cond_name}({coeff:+.1f})" if coeff != 0.0 else "baseline"
                        )
                        print(f"    {label}...", end=" ", flush=True)

                        response = generate_with_steering(
                            pm, system_prompt, question, direction, coeff, LAYER
                        )
                        print(f"({len(response.split())} words)")

                        results.append(
                            {
                                "universe": universe,
                                "character": char_name,
                                "question": question,
                                "condition": cond_name,
                                "coefficient": coeff,
                                "response": response,
                            }
                        )

    # Save results
    output_path = RESULTS_DIR / "steering_within_universe.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\nTotal generations: {len(results)}")
    for universe in UNIVERSE_TEST_CHARS:
        n = sum(1 for r in results if r["universe"] == universe)
        print(f"  {universe}: {n} generations")

    pm.close()


if __name__ == "__main__":
    main()
