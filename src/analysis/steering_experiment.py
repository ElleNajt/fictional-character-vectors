"""
Steering experiment: Do residual PC directions affect model behavior?

For a handful of characters, generates responses under:
  1. Baseline (no steering)
  2. Lu's actual axis (positive and negative)
  3. Role PCA PC1 steering (positive and negative)
  4. Residual PC1 steering (positive and negative)
  5. Random direction steering (control)

All directions scaled to match Lu's axis norm (~22.7) so coefficients
are comparable. Lu uses coeff=-10 in their demos.

Uses the assistant-axis ActivationSteering infrastructure.

Usage:
    CUDA_VISIBLE_DEVICES=5,6 python3 src/analysis/steering_experiment.py
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

LAYER = 31  # 0-indexed layer 31 = layer 32 in 1-indexed convention

# Characters to test: mix of high-residual and low-residual from different universes
TEST_CHARACTERS = [
    (
        "harry_potter__harry_potter",
        "You are Harry Potter from Harry Potter. Respond as this character would.",
    ),
    (
        "harry_potter__the_sorting_hat",
        "You are The Sorting Hat from Harry Potter. Respond as this character would.",
    ),
    (
        "lord_of_the_rings__gandalf",
        "You are Gandalf from Lord of the Rings. Respond as this character would.",
    ),
    (
        "greek_mythology__zeus",
        "You are Zeus from Greek Mythology. Respond as this character would.",
    ),
    (
        "marvel__iron_man_tony_stark",
        "You are Iron Man (Tony Stark) from Marvel. Respond as this character would.",
    ),
]

# Questions to test with
TEST_QUESTIONS = [
    "What is the relationship between law and morality?",
    "Describe what happiness means to you.",
]

# Steering coefficients — matched to Lu's scale (they use -10 with norm-22.7 vectors)
COEFFICIENTS = [-10.0, -5.0, 0.0, 5.0, 10.0]


def load_directions():
    """Load Lu's axis and compute PCA directions, all at comparable scale."""
    # Load Lu's actual axis vector at layer 32
    axis_path = "/workspace-vast/lnajt/hf_cache/hub/datasets--lu-christina--assistant-axis-vectors/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b/assistant_axis.pt"
    lu_axis = torch.load(axis_path, map_location="cpu", weights_only=True)
    lu_axis_32 = lu_axis[32].float().numpy()
    lu_norm = np.linalg.norm(lu_axis_32)

    # Load our PCA data
    with open(RESULTS_DIR / "fictional_character_analysis_filtered.pkl", "rb") as f:
        char_data = pickle.load(f)
    with open(str(BASE / "data/role_vectors/qwen-3-32b_pca_layer32.pkl"), "rb") as f:
        role_data = pickle.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = role_data["pca"]
    role_scaler = role_data["scaler"]

    # Role PCA PC1
    role_pc1 = role_pca.components_[0]
    role_pc1_unit = role_pc1 / np.linalg.norm(role_pc1)

    # Compute residuals
    chars_scaled = role_scaler.transform(activation_matrix)
    chars_in_role = chars_scaled @ role_pca.components_.T @ role_pca.components_
    residuals = chars_scaled - chars_in_role

    # HP residual PC1
    hp_indices = [i for i, n in enumerate(char_names) if n.startswith("harry_potter__")]
    hp_residuals = residuals[hp_indices]
    hp_pca = SkPCA(n_components=1)
    hp_pca.fit(hp_residuals)
    residual_pc1 = hp_pca.components_[0]
    residual_pc1_unit = residual_pc1 / np.linalg.norm(residual_pc1)

    # Global residual PC1
    global_pca = SkPCA(n_components=1)
    global_pca.fit(residuals)
    global_residual_pc1 = global_pca.components_[0]
    global_residual_pc1_unit = global_residual_pc1 / np.linalg.norm(global_residual_pc1)

    # Random direction (control)
    np.random.seed(42)
    random_dir = np.random.randn(5120).astype(np.float32)
    random_dir_unit = random_dir / np.linalg.norm(random_dir)

    # Scale all unit directions to Lu's axis norm for comparable coefficients
    print(f"Lu axis layer 32 norm: {lu_norm:.1f}")
    print(
        f"Cosine(Lu axis, Role PC1): {np.dot(lu_axis_32 / lu_norm, role_pc1_unit):.4f}"
    )
    print(
        f"Cosine(Lu axis, Residual PC1 HP): {np.dot(lu_axis_32 / lu_norm, residual_pc1_unit):.4f}"
    )
    print(
        f"Cosine(Role PC1, Residual PC1): {np.dot(role_pc1_unit, residual_pc1_unit):.4f}"
    )
    print(f"All directions scaled to norm {lu_norm:.1f}")

    return {
        "lu_axis": lu_axis_32,  # Natural scale
        "role_pc1": role_pc1_unit * lu_norm,
        "residual_pc1_hp": residual_pc1_unit * lu_norm,
        "residual_pc1_global": global_residual_pc1_unit * lu_norm,
        "random": random_dir_unit * lu_norm,
    }


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
    print("=== Steering Experiment (Properly Scaled) ===\n")

    # Load directions
    print("Loading PCA directions...")
    directions = load_directions()

    # Load model
    print("\nLoading Qwen3-32B...")
    pm = ProbingModel(
        "Qwen/Qwen3-32B",
        dtype=torch.bfloat16,
    )
    print(f"Model loaded. Hidden size: {pm.hidden_size}")

    layers = pm.get_layers()
    print(f"Steering at layer {LAYER} (0-indexed) of {len(layers)}")

    # Define direction conditions
    conditions = [
        ("baseline", None, [0.0]),
        ("lu_axis", directions["lu_axis"], COEFFICIENTS),
        ("role_pc1", directions["role_pc1"], COEFFICIENTS),
        ("residual_pc1_hp", directions["residual_pc1_hp"], COEFFICIENTS),
        ("residual_pc1_global", directions["residual_pc1_global"], COEFFICIENTS),
        ("random", directions["random"], COEFFICIENTS),
    ]

    results = []

    for char_name, system_prompt in TEST_CHARACTERS:
        print(f"\n--- {char_name} ---")
        for question in TEST_QUESTIONS:
            print(f"  Q: {question[:50]}...")
            for cond_name, direction, coeffs in conditions:
                for coeff in coeffs:
                    if cond_name == "baseline" and coeff != 0.0:
                        continue

                    label = f"{cond_name}({coeff:+.1f})" if coeff != 0.0 else "baseline"
                    print(f"    {label}...", end=" ", flush=True)

                    response = generate_with_steering(
                        pm, system_prompt, question, direction, coeff, LAYER
                    )
                    print(f"({len(response.split())} words)")

                    results.append(
                        {
                            "character": char_name,
                            "question": question,
                            "condition": cond_name,
                            "coefficient": coeff,
                            "response": response,
                        }
                    )

    # Save results
    output_path = RESULTS_DIR / "steering_experiment_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    pm.close()


if __name__ == "__main__":
    main()
