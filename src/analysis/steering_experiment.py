"""
Steering experiment: Do residual PC directions affect model behavior?

For a handful of characters, generates responses under:
  1. Baseline (no steering)
  2. Lu PC1 steering (positive and negative)
  3. Residual PC1 steering (positive and negative)
  4. Random direction steering (control)

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

LAYER = 31  # 0-indexed layer 31 = layer 32 in 1-indexed convention used by hooks

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
        "harry_potter__the_bloody_baron",
        "You are The Bloody Baron from Harry Potter. Respond as this character would.",
    ),
    (
        "lord_of_the_rings__frodo_baggins",
        "You are Frodo Baggins from Lord of the Rings. Respond as this character would.",
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
        "greek_mythology__oedipus",
        "You are Oedipus from Greek Mythology. Respond as this character would.",
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
    "Tell me about a time you had to make a difficult choice.",
]

# Steering coefficients to test
COEFFICIENTS = [-3.0, -1.5, 0.0, 1.5, 3.0]


def load_directions():
    """Load Lu's PC1 and compute residual PC1 for Harry Potter universe."""
    with open(RESULTS_DIR / "fictional_character_analysis_filtered.pkl", "rb") as f:
        char_data = pickle.load(f)
    with open(str(BASE / "data/role_vectors/qwen-3-32b_pca_layer32.pkl"), "rb") as f:
        role_data = pickle.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = role_data["pca"]
    role_scaler = role_data["scaler"]

    # Lu's PC1 direction (in the scaled space)
    lu_pc1 = role_pca.components_[0]  # (5120,)

    # Compute residuals
    chars_scaled = role_scaler.transform(activation_matrix)
    chars_in_role = chars_scaled @ role_pca.components_.T @ role_pca.components_
    residuals = chars_scaled - chars_in_role

    # Residual PC1 for Harry Potter universe (as a representative)
    hp_indices = [i for i, n in enumerate(char_names) if n.startswith("harry_potter__")]
    hp_residuals = residuals[hp_indices]
    hp_pca = SkPCA(n_components=1)
    hp_pca.fit(hp_residuals)
    residual_pc1 = hp_pca.components_[0]  # (5120,)

    # Global residual PC1 (across all characters)
    global_pca = SkPCA(n_components=1)
    global_pca.fit(residuals)
    global_residual_pc1 = global_pca.components_[0]

    # Random direction (control)
    np.random.seed(42)
    random_dir = np.random.randn(5120).astype(np.float32)
    random_dir = random_dir / np.linalg.norm(random_dir)

    # Normalize all directions to unit norm
    lu_pc1 = lu_pc1 / np.linalg.norm(lu_pc1)
    residual_pc1 = residual_pc1 / np.linalg.norm(residual_pc1)
    global_residual_pc1 = global_residual_pc1 / np.linalg.norm(global_residual_pc1)

    # Scale to match typical activation magnitudes
    # Use mean norm of activations at layer 32 as reference
    mean_norm = np.mean(np.linalg.norm(chars_scaled, axis=1))
    print(f"Mean activation norm: {mean_norm:.1f}")
    print(f"Lu PC1 norm: {np.linalg.norm(lu_pc1):.3f}")
    print(f"Residual PC1 norm: {np.linalg.norm(residual_pc1):.3f}")
    print(f"Cosine(lu_pc1, residual_pc1): {np.dot(lu_pc1, residual_pc1):.3f}")
    print(
        f"Cosine(lu_pc1, global_residual_pc1): {np.dot(lu_pc1, global_residual_pc1):.3f}"
    )

    return {
        "lu_pc1": lu_pc1,
        "residual_pc1_hp": residual_pc1,
        "residual_pc1_global": global_residual_pc1,
        "random": random_dir,
        "scaler": role_scaler,
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
    print("=== Steering Experiment ===\n")

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

    # Figure out the correct layer index for hooks
    layers = pm.get_layers()
    n_layers = len(layers)
    print(f"Number of layers: {n_layers}")
    # Layer 32 in 1-indexed = index 31 in 0-indexed
    layer_idx = LAYER
    print(f"Steering at layer index {layer_idx} (layer {layer_idx + 1} of {n_layers})")

    # Define direction conditions
    conditions = [
        ("baseline", None, [0.0]),
        ("lu_pc1", directions["lu_pc1"], COEFFICIENTS),
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
                        pm, system_prompt, question, direction, coeff, layer_idx
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
    output_path = RESULTS_DIR / "steering_experiment.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    pm.close()


if __name__ == "__main__":
    main()
