"""Logit lens interpretation of PCA axes.

Projects the model's unembedding matrix (LM head) onto each universe's
PC1 direction to find which tokens are most associated with each end
of the axis.

The idea: at layer 32, the residual stream will eventually be linearly
decoded into logits via the LM head. So projecting the LM head rows
(one per vocab token) onto a direction in activation space tells you
which tokens that direction "points toward."

Runs two modes:
  1. Residual PCA (after removing Lu's role space)
  2. Raw PCA (no role-space removal)

Filters garbage tokens (code fragments, byte sequences, control chars)
while keeping meaningful English and Chinese tokens.

Outputs:
  results/logit_lens_pca.txt      - residual PC logit lens
  results/logit_lens_pca_raw.txt  - raw PC logit lens
"""

import json
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "Qwen/Qwen3-32B"


def is_meaningful_token(token_str):
    """Filter out garbage tokens (code, byte sequences, control chars).

    Keeps: real English words, Chinese characters, common punctuation.
    Removes: code fragments, hex/byte sequences, control characters,
             lone subword fragments (e.g. 'ĠĠĠ'), HTML tags, etc.
    """
    s = token_str.strip()
    if not s:
        return False
    # Remove tokens that are pure whitespace / special whitespace chars
    if all(c in " \t\n\r\x00" or unicodedata.category(c).startswith("Z") for c in s):
        return False
    # Remove byte-level tokens like <0x0A>
    if re.match(r"^<0x[0-9A-Fa-f]+>$", s):
        return False
    # Remove tokens that are mostly non-printable or control chars
    printable = sum(1 for c in s if unicodedata.category(c)[0] not in ("C",))
    if printable < len(s) * 0.5:
        return False
    # Remove tokens with too many special/code characters (braces, backslashes, etc.)
    code_chars = sum(1 for c in s if c in "{}[]\\|<>@#$%^&*~`")
    if code_chars > len(s) * 0.3:
        return False
    # Remove tokens that look like hex/hash sequences
    if re.match(r"^[0-9a-fA-F]{6,}$", s):
        return False
    # Remove pure digit strings (not meaningful)
    if re.match(r"^\d+$", s):
        return False
    # Remove tokens that are just repeated special chars (Ġ, ĉ, etc.)
    unique_chars = set(s)
    if len(unique_chars) <= 2 and not any(
        unicodedata.category(c)[0] in ("L",) for c in unique_chars
    ):
        return False
    return True


# --- Load character data and compute residual PCs ---
print("Loading character data...")
with open("results/fictional_character_analysis_filtered.pkl", "rb") as f:
    char_data = pickle.load(f)

char_names = char_data["character_names"]
activation_matrix = char_data["activation_matrix"]

with open("data/role_vectors/qwen-3-32b_pca_layer32.pkl", "rb") as f:
    lu_data = pickle.load(f)
role_pca = lu_data["pca"]
lu_scaler = lu_data["scaler"]
lu_role_names = lu_data["role_names"]

scaler = StandardScaler()
chars_scaled = scaler.fit_transform(activation_matrix)
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


def get_universe_indices(prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [
        i
        for i, name in enumerate(char_names)
        if any(name.startswith(p) for p in prefixes)
    ]


def compute_universe_pcs(data_matrix, label):
    """Compute per-universe PC1 directions from a data matrix (scaled space)."""
    universe_pcs = {}
    print(f"Computing {label} PC1 directions...")
    for universe, prefixes in ALL_UNIVERSES.items():
        indices = get_universe_indices(prefixes)
        if len(indices) < 20:
            continue
        u_data = data_matrix[indices]
        u_names = [char_names[i] for i in indices]

        u_pca = SkPCA(n_components=1)
        scores = u_pca.fit_transform(u_data)[:, 0]
        pc1_dir = u_pca.components_[0]  # (5120,)

        sorted_idx = np.argsort(scores)
        high_chars = [
            u_names[i].split("__")[-1].replace("_", " ").title()
            for i in sorted_idx[-5:][::-1]
        ]
        low_chars = [
            u_names[i].split("__")[-1].replace("_", " ").title() for i in sorted_idx[:5]
        ]

        universe_pcs[universe] = {
            "direction": pc1_dir,
            "var_explained": u_pca.explained_variance_ratio_[0],
            "high_chars": high_chars,
            "low_chars": low_chars,
        }
        print(f"  {universe}: PC1 explains {u_pca.explained_variance_ratio_[0]:.1%}")

    # Global PCs
    global_pca = SkPCA(n_components=5)
    global_pca.fit(data_matrix)
    for pc_idx in range(3):
        universe_pcs[f"Global {label} PC{pc_idx + 1}"] = {
            "direction": global_pca.components_[pc_idx],
            "var_explained": global_pca.explained_variance_ratio_[pc_idx],
            "high_chars": [],
            "low_chars": [],
        }
    return universe_pcs


# Compute both residual and raw PC directions
residual_pcs = compute_universe_pcs(residuals, "Residual")
raw_pcs = compute_universe_pcs(chars_scaled, "Raw")

# --- Lu's role PCA directions ---
# These are the generic "role" axes from Lu et al., fit on 275 role vectors.
# Components live in Lu's scaled space, so we unscale with lu_scaler.
print("\nBuilding Lu role PCA directions...")
N_LU_PCS = 10

# Project our characters into Lu's space to find who's extreme on each Lu PC.
chars_lu_scaled = lu_scaler.transform(activation_matrix)
chars_in_lu_pcs = role_pca.transform(chars_lu_scaled)  # (n_chars, 275)

lu_pcs = {}
for pc_idx in range(N_LU_PCS):
    scores = chars_in_lu_pcs[:, pc_idx]
    sorted_idx = np.argsort(scores)
    high_chars = [
        char_names[i].split("__")[-1].replace("_", " ").title()
        for i in sorted_idx[-5:][::-1]
    ]
    low_chars = [
        char_names[i].split("__")[-1].replace("_", " ").title() for i in sorted_idx[:5]
    ]
    lu_pcs[f"Lu Role PC{pc_idx + 1}"] = {
        "direction": role_pca.components_[pc_idx],
        "var_explained": role_pca.explained_variance_ratio_[pc_idx],
        "high_chars": high_chars,
        "low_chars": low_chars,
    }
    print(
        f"  Lu Role PC{pc_idx + 1}: explains {role_pca.explained_variance_ratio_[pc_idx]:.1%}"
    )

# --- Load model LM head ---
# The LM head maps (5120,) -> (vocab_size,) via W @ x
# We need the weight matrix W: (vocab_size, 5120)
# But we work in *scaled* space (StandardScaler), so we need to account for that.
#
# The PC direction d is in scaled space. The scaler does: x_scaled = (x - mean) / std
# So the unscaling is: x_original = x_scaled * std + mean
# A direction in scaled space corresponds to d_original = d * std in original space.
# (The mean shift doesn't affect directions, only the bias.)
#
# The LM head operates on original (unscaled) activations, so we project:
#   lm_head @ (d * std) = lm_head @ d_unscaled
#
# Actually, we also need to account for the role projection removal:
#   residual_scaled = x_scaled - (x_scaled @ role_components^T) @ role_components
# The PC direction is in this residual-scaled space.
#
# For the logit lens, the cleanest approach is to just use the direction as-is
# in the scaled space, unscale it, then project the LM head onto it.
# The direction in original activation space is: d_orig = d_scaled * scaler.scale_

print("\nLoading model LM head weight only...")
# Load just the lm_head.weight from safetensors, not the full model.
# This avoids loading 60GB+ of weights into RAM.
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Find which shard contains lm_head.weight
index_path = hf_hub_download(MODEL_NAME, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)
shard_file = index["weight_map"]["lm_head.weight"]
shard_path = hf_hub_download(MODEL_NAME, shard_file)

with safe_open(shard_path, framework="pt") as f:
    lm_head_weight = f.get_tensor("lm_head.weight").float().numpy()

print(f"LM head shape: {lm_head_weight.shape}")

# --- Build token filter mask ---
print("\nBuilding token filter...")
vocab_size = lm_head_weight.shape[0]
token_strs = [tokenizer.decode([i]) for i in range(vocab_size)]
meaningful_mask = np.array([is_meaningful_token(s) for s in token_strs])
n_kept = meaningful_mask.sum()
print(f"Keeping {n_kept}/{vocab_size} tokens ({n_kept / vocab_size:.1%})")


def run_logit_lens(universe_pcs, mode_label, output_path, scale=None):
    """Project LM head onto PC directions and write results.

    scale: per-feature std to unscale directions. Defaults to scaler.scale_.
    """
    if scale is None:
        scale = scaler.scale_

    print(f"\n{'#' * 70}")
    print(f"# {mode_label}")
    print(f"{'#' * 70}")

    out = [f"# {mode_label}\n"]
    TOP_K = 30

    for universe, data in universe_pcs.items():
        d_scaled = data["direction"]  # (5120,) in scaled space

        # Unscale: direction in original activation space
        d_orig = d_scaled * scale
        d_orig_norm = d_orig / np.linalg.norm(d_orig)

        # Project each vocab token's LM head row onto the direction
        projections = lm_head_weight @ d_orig_norm  # (vocab_size,)

        # Apply filter: set garbage tokens to 0 so they don't appear in ranking
        filtered_proj = projections.copy()
        filtered_proj[~meaningful_mask] = 0.0

        sorted_idx = np.argsort(filtered_proj)

        header = f"\n{'=' * 70}\n{universe}"
        if data["high_chars"]:
            header += f"\n  HIGH chars: {', '.join(data['high_chars'])}"
            header += f"\n  LOW chars:  {', '.join(data['low_chars'])}"
        header += f"\n  Variance explained: {data['var_explained']:.1%}"
        header += f"\n{'=' * 70}"
        out.append(header)
        print(header)

        for end_label, indices in [
            (f"Top {TOP_K} tokens (HIGH end)", sorted_idx[-TOP_K:][::-1]),
            (f"Bottom {TOP_K} tokens (LOW end)", sorted_idx[:TOP_K]),
        ]:
            out.append(f"\n  {end_label}:")
            print(f"\n  {end_label}:")
            for i in indices:
                score = projections[i]  # original (unfiltered) score
                line = f"    {score:+8.3f}  id={i:6d}  '{token_strs[i]}'"
                out.append(line)
                print(line)

    Path(output_path).write_text("\n".join(out))
    print(f"\nSaved {output_path}")


run_logit_lens(
    residual_pcs,
    "RESIDUAL PCA (after role-space removal)",
    "results/logit_lens_pca.txt",
)
run_logit_lens(
    raw_pcs, "RAW PCA (no role-space removal)", "results/logit_lens_pca_raw.txt"
)
run_logit_lens(
    lu_pcs,
    "LU ROLE PCA (275 generic roles)",
    "results/logit_lens_pca_lu.txt",
    scale=lu_scaler.scale_,
)
