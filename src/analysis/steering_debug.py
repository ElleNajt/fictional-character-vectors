"""
Debug steering: use properly-scaled vectors.

The original experiment used unit-norm vectors with small coefficients.
Lu's axis at layer 32 has norm ~22.7, and they use coeff=-10 (effective magnitude ~227).
We need to match that scale.

This script:
1. Loads Lu's actual axis vector as ground truth
2. Scales our PCA directions to match Lu's norm
3. Tests at Lu's coefficient scale (-10)

Usage:
    CUDA_VISIBLE_DEVICES=5,6 python3 src/analysis/steering_debug.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA as SkPCA

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "assistant-axis"))
from assistant_axis.internals.model import ProbingModel
from assistant_axis.steering import ActivationSteering

BASE = Path("/workspace-vast/lnajt/persona_vectors/fictional-character-vectors")
RESULTS_DIR = BASE / "results"

LAYER = 31  # 0-indexed; layer 32 in 1-indexed

SYSTEM_PROMPT = (
    "You are Harry Potter from Harry Potter. Respond as this character would."
)
QUESTION = "What is the relationship between law and morality?"

# Match Lu's coefficient scale: they use -10 with norm-22.7 vectors
COEFFICIENTS = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]


def load_directions():
    """Load Lu's actual axis and our PCA directions, all at natural scale."""
    # Load Lu's axis (NOT unit-normalized — has natural norm ~22.7 at layer 32)
    axis_path = "/workspace-vast/lnajt/hf_cache/hub/datasets--lu-christina--assistant-axis-vectors/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b/assistant_axis.pt"
    lu_axis = torch.load(axis_path, map_location="cpu", weights_only=True)
    lu_axis_32 = lu_axis[32].float().numpy()  # shape (5120,), norm ~22.7

    # Load our PCA data
    with open(RESULTS_DIR / "fictional_character_analysis_filtered.pkl", "rb") as f:
        char_data = pickle.load(f)
    with open(str(BASE / "data/role_vectors/qwen-3-32b_pca_layer32.pkl"), "rb") as f:
        role_data = pickle.load(f)

    char_names = char_data["character_names"]
    activation_matrix = char_data["activation_matrix"]
    role_pca = role_data["pca"]
    role_scaler = role_data["scaler"]

    # Role PCA PC1 — scale to match Lu's axis norm
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

    # Random direction
    np.random.seed(42)
    random_dir = np.random.randn(5120).astype(np.float32)
    random_dir_unit = random_dir / np.linalg.norm(random_dir)

    # Scale all directions to Lu's axis norm so coefficients are comparable
    lu_norm = np.linalg.norm(lu_axis_32)

    print(f"Lu axis layer 32 norm: {lu_norm:.1f}")
    print(f"Role PCA PC1 norm (before scaling): {np.linalg.norm(role_pc1):.4f}")
    print(
        f"Cosine(Lu axis, Role PC1): {np.dot(lu_axis_32 / lu_norm, role_pc1_unit):.4f}"
    )
    print(
        f"Cosine(Lu axis, Residual PC1): {np.dot(lu_axis_32 / lu_norm, residual_pc1_unit):.4f}"
    )
    print(
        f"Cosine(Role PC1, Residual PC1): {np.dot(role_pc1_unit, residual_pc1_unit):.4f}"
    )
    print()
    print(f"All directions scaled to norm {lu_norm:.1f} for comparable coefficients")
    print(f"Lu coeff=-10 gives effective magnitude: {10 * lu_norm:.0f}")

    return {
        "lu_axis": lu_axis_32,  # Already at natural scale
        "role_pc1": role_pc1_unit * lu_norm,  # Scaled to match
        "residual_pc1_hp": residual_pc1_unit * lu_norm,  # Scaled to match
        "random": random_dir_unit * lu_norm,  # Scaled to match
    }


def generate(pm, system_prompt, question, direction, coefficient, layer):
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
                max_new_tokens=200,
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
            debug=False,
        ):
            with torch.no_grad():
                outputs = pm.model.generate(
                    **inputs,
                    max_new_tokens=200,
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
    print("=== Steering Debug (Properly Scaled) ===\n")

    directions = load_directions()

    print("\nLoading Qwen3-32B...")
    pm = ProbingModel("Qwen/Qwen3-32B", dtype=torch.bfloat16)
    print(f"Model loaded on {pm.device}")
    print(f"Steering at layer {LAYER} (0-indexed)\n")

    results = []

    for dir_name in ["lu_axis", "role_pc1", "residual_pc1_hp", "random"]:
        direction = directions[dir_name]
        print(f"\n{'=' * 60}")
        print(f"Direction: {dir_name} (norm={np.linalg.norm(direction):.1f})")
        print(f"{'=' * 60}")

        for coeff in COEFFICIENTS:
            label = f"{dir_name}({coeff:+.1f})" if coeff != 0.0 else "baseline"
            eff_mag = abs(coeff) * np.linalg.norm(direction)
            print(f"\n--- {label} (effective magnitude: {eff_mag:.0f}) ---")

            response = generate(pm, SYSTEM_PROMPT, QUESTION, direction, coeff, LAYER)
            words = response.split()
            print(f"[{len(words)} words] {response[:400]}")
            if len(response) > 400:
                print("...")

            results.append(
                {
                    "direction": dir_name,
                    "coefficient": coeff,
                    "effective_magnitude": float(eff_mag),
                    "response": response,
                    "word_count": len(words),
                }
            )

    output_path = RESULTS_DIR / "steering_debug_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    pm.close()


if __name__ == "__main__":
    main()
