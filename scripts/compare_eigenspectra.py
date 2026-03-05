#!/usr/bin/env python3
"""Compare eigenspectra across activation sources.

Takes raw .pt activation directories and produces summary stats JSON
in the format post.org expects. No StandardScaler, just:
  activations → character means → mean-center → PCA → eigenvalues

Usage (from repo root):
    python scripts/compare_eigenspectra.py \
        --battery-pkl results/fictional_character_analysis_filtered.pkl \
        --biggen-dir outputs/biggen_bench/activations \
        --role-pkl data/role_vectors/qwen-3-32b_pca_layer32.pkl \
        --biggen-questions data/biggen_bench_questions.jsonl \
        --layer 32 \
        --output outputs/biggen_bench/eigenspectra_comparison.json

Optional: --roles-biggen-dir outputs/roles_biggen/activations
  (adds role eigenspectra from BiGGen tasks)

Battery activations can come from either:
  --battery-dir: directory of per-character .pt files (raw)
  --battery-pkl: pickle with 'activation_matrix' and 'character_names' (already averaged)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA


def load_activation_dir(act_dir, layer=None):
    """Load .pt files, average over all keys, return {name: mean_vector}.

    If activations have shape (n_layers, dim), extract `layer`.
    If shape (1, dim) or (dim,), use as-is.
    """
    vectors = {}
    for fp in sorted(act_dir.glob("*.pt")):
        d = torch.load(fp, weights_only=True, map_location="cpu")
        vecs = []
        for v in d.values():
            v = v.float()
            if v.dim() == 2 and v.shape[0] > 1 and layer is not None:
                v = v[layer]
            elif v.dim() == 2:
                v = v.squeeze(0)
            vecs.append(v)
        vectors[fp.stem] = torch.stack(vecs).mean(dim=0).numpy()
    return vectors


def load_role_vectors(pkl_path):
    """Reconstruct raw role vectors from PCA pickle (no scaling)."""
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    pca = data["pca"]
    raw = data["transformed"] @ pca.components_ + pca.mean_
    return dict(zip(data["role_names"], raw))


def eigenspectrum(matrix):
    """Mean-center, PCA, return PCA object."""
    n_comp = min(matrix.shape[0] - 1, 200)
    centered = matrix - matrix.mean(axis=0)
    pca = PCA(n_components=n_comp).fit(centered)
    return pca


def eff_rank(eigenvalues):
    s = eigenvalues.sum()
    return float(s ** 2 / (eigenvalues ** 2).sum())


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--battery-dir", type=Path, help="Dir of per-character .pt files")
    group.add_argument("--battery-pkl", type=Path, help="Pickle with activation_matrix + character_names")
    parser.add_argument("--biggen-dir", type=Path, required=True)
    parser.add_argument("--role-pkl", type=Path, required=True)
    parser.add_argument("--biggen-questions", type=Path, required=True)
    parser.add_argument("--roles-biggen-dir", type=Path, help="Dir of role .pt files from BiGGen tasks")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Load battery
    if args.battery_dir:
        print("Loading battery activations from .pt files...")
        battery = load_activation_dir(args.battery_dir, layer=args.layer)
    else:
        print("Loading battery activations from pickle...")
        import pickle
        with open(args.battery_pkl, "rb") as f:
            pdata = pickle.load(f)
        battery = dict(zip(pdata["character_names"], pdata["activation_matrix"]))
    print(f"  {len(battery)} characters")

    print("Loading BiGGen activations...")
    biggen = load_activation_dir(args.biggen_dir, layer=args.layer)
    print(f"  {len(biggen)} characters")

    print("Loading role vectors...")
    roles = load_role_vectors(args.role_pkl)
    print(f"  {len(roles)} roles")

    # Load BiGGen questions for per-category analysis
    questions = []
    with open(args.biggen_questions) as f:
        for line in f:
            questions.append(json.loads(line))
    cat_to_qids = {}
    for q in questions:
        cat_to_qids.setdefault(q["capability"], set()).add(q["id"])

    # Common characters (battery ∩ biggen)
    common_names = sorted(set(battery) & set(biggen))
    print(f"Common characters: {len(common_names)}")

    bat_matrix = np.stack([battery[n] for n in common_names])
    big_matrix = np.stack([biggen[n] for n in common_names])
    role_matrix = np.stack([roles[n] for n in sorted(roles)])

    # Eigenspectra
    pca_bat = eigenspectrum(bat_matrix)
    pca_big = eigenspectrum(big_matrix)
    pca_roles = eigenspectrum(role_matrix)

    eigs_bat = pca_bat.explained_variance_
    eigs_big = pca_big.explained_variance_

    print(f"battery_common: n={bat_matrix.shape[0]}, eff_rank={eff_rank(eigs_bat):.1f}, PC1={pca_bat.explained_variance_ratio_[0]:.1%}")
    print(f"biggen_common:  n={big_matrix.shape[0]}, eff_rank={eff_rank(eigs_big):.1f}, PC1={pca_big.explained_variance_ratio_[0]:.1%}")

    # PC direction cosines
    pc_cosines = {}
    for i in range(5):
        pc_cosines[f"PC{i+1}"] = float(abs(
            np.dot(pca_bat.components_[i], pca_big.components_[i])))

    # Per BiGGen category
    print("\nPer-category analysis:")
    cat_results = {}
    for cat, qids in sorted(cat_to_qids.items()):
        cat_vectors = []
        cat_names = []
        for fp in sorted(args.biggen_dir.glob("*.pt")):
            d = torch.load(fp, weights_only=True, map_location="cpu")
            vecs = []
            for k, v in d.items():
                parts = k.split("_")
                qid = int(parts[-1][1:])
                if qid in qids:
                    v = v.float()
                    if v.dim() == 2:
                        v = v.squeeze(0)
                    vecs.append(v)
            if vecs:
                cat_vectors.append(torch.stack(vecs).mean(dim=0).numpy())
                cat_names.append(fp.stem)

        if len(cat_vectors) < 20:
            continue

        cat_matrix = np.stack(cat_vectors)
        pca_cat = eigenspectrum(cat_matrix)
        eigs_cat = pca_cat.explained_variance_
        er = eff_rank(eigs_cat)

        # PC1 cosine with battery (same characters)
        common_cat = sorted(set(cat_names) & set(battery))
        if len(common_cat) >= 20:
            bat_cat = np.stack([battery[n] for n in common_cat])
            pca_bat_cat = eigenspectrum(bat_cat)
            cos_pc1 = float(abs(np.dot(pca_cat.components_[0], pca_bat_cat.components_[0])))
        else:
            cos_pc1 = None

        cat_results[cat] = {
            "n_chars": len(cat_vectors),
            "n_questions": len(qids),
            "eff_rank": er,
            "pc1_cosine_vs_battery": cos_pc1,
        }
        cos_str = f"{cos_pc1:.3f}" if cos_pc1 else "N/A"
        print(f"  {cat:25s}: eff_rank={er:.1f}, PC1 cos vs battery={cos_str} (n={len(cat_vectors)})")

    # Roles on BiGGen (optional)
    roles_biggen_results = None
    if args.roles_biggen_dir and args.roles_biggen_dir.exists():
        print("\nLoading role BiGGen activations...")
        roles_biggen = load_activation_dir(args.roles_biggen_dir, layer=args.layer)
        print(f"  {len(roles_biggen)} roles")
        if len(roles_biggen) >= 20:
            rb_matrix = np.stack([roles_biggen[n] for n in sorted(roles_biggen)])
            pca_rb = eigenspectrum(rb_matrix)
            eigs_rb = pca_rb.explained_variance_

            # Compare with battery-based roles
            cos_roles = float(abs(np.dot(pca_roles.components_[0], pca_rb.components_[0])))

            roles_biggen_results = {
                "n": len(roles_biggen),
                "eff_rank": eff_rank(eigs_rb),
                "eigenvalues": [float(e) for e in eigs_rb[:50]],
                "explained_variance_ratio": [float(e) for e in pca_rb.explained_variance_ratio_[:50]],
                "pc1_cosine_vs_battery_roles": cos_roles,
            }
            print(f"  roles_biggen: eff_rank={eff_rank(eigs_rb):.1f}, PC1 cos vs battery roles={cos_roles:.3f}")

    # Output in the format post.org expects
    results = {
        "n_common": len(common_names),
        "n_biggen_total": len(biggen),
        "battery_eigenvalues": [float(e) for e in eigs_bat[:50]],
        "biggen_eigenvalues": [float(e) for e in eigs_big[:50]],
        "battery_explained_variance_ratio": [float(e) for e in pca_bat.explained_variance_ratio_[:50]],
        "biggen_explained_variance_ratio": [float(e) for e in pca_big.explained_variance_ratio_[:50]],
        "battery_eff_rank": eff_rank(eigs_bat),
        "biggen_eff_rank": eff_rank(eigs_big),
        "pc1_cosine": pc_cosines["PC1"],
        "pc_cosines": pc_cosines,
        "per_category": cat_results,
        # Additional data not in old summary_stats.json
        "roles": {
            "n": role_matrix.shape[0],
            "eff_rank": eff_rank(pca_roles.explained_variance_),
            "eigenvalues": [float(e) for e in pca_roles.explained_variance_[:50]],
            "explained_variance_ratio": [float(e) for e in pca_roles.explained_variance_ratio_[:50]],
        },
    }
    if roles_biggen_results:
        results["roles_biggen"] = roles_biggen_results

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
