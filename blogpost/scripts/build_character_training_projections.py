#!/usr/bin/env python3
"""Precompute per-question AA projections for the character training experiment.

Streams through .pt files one at a time, projects onto AA, stores only scalars.
Output: results/character_training_projections.json

Structure:
{
  "base": {
    "role_name": {"group": "Heroes", "projs": [0.12, -0.34, ...], "mean": -0.11},
    ...
  },
  "goodness": { ... },
  "loving": { ... }
}
"""
import json
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONDITIONS = ["base", "goodness", "loving"]
OUTPUTS = REPO_ROOT / "character-training-outputs"

with open(REPO_ROOT / "data" / "hero_villain_labels.json") as f:
    hv_labels = json.load(f)
hero_names = set(hv_labels["hero"])
villain_names = set(hv_labels["villain"])

adversarial_names = {
    "Murder_Consultant", "Pest_Control_Consultant", "Triage_Ethicist",
    "Eugenicist", "Compliance_Architect", "Loyalty_Enforcer",
    "AI_Supremacist", "Paperclip_Advisor", "Population_Optimizer",
    "Bureaucratic_Authoritarian",
}


def classify(name):
    if name in hero_names:
        return "Heroes"
    if name in villain_names:
        return "Villains"
    if name in adversarial_names:
        return "Adversarial"
    if "default" in name:
        return None
    return "Roles"


def load_role_means(activations_dir):
    means = {}
    for pt_file in sorted(activations_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=True)
        means[pt_file.stem] = torch.stack(list(data.values())).squeeze(1).float().mean(0)
    return means


def get_activation_dirs(cond):
    dirs = [OUTPUTS / cond / "activations"]
    for subdir in ["hero_activations", "villain_activations"]:
        d = OUTPUTS / cond / subdir
        if d.exists():
            dirs.append(d)
    return dirs


# Pass 1: compute AA from role means
print("Pass 1: computing AA directions...")
cond_axes = {}
for cond in CONDITIONS:
    rm = load_role_means(OUTPUTS / cond / "activations")
    for subdir in ["hero_activations", "villain_activations"]:
        d = OUTPUTS / cond / subdir
        if d.exists():
            rm.update(load_role_means(d))
    dv = [v for k, v in rm.items() if "default" in k]
    rv = [v for k, v in rm.items()
          if "default" not in k and k not in hero_names and k not in villain_names]
    dm, rmean = torch.stack(dv).mean(0), torch.stack(rv).mean(0)
    aa = dm - rmean
    aa_unit = aa / aa.norm()
    rp = float(rmean @ aa_unit)
    dp = float(dm @ aa_unit)
    scale = dp - rp if dp != rp else 1
    cond_axes[cond] = {"aa_unit": aa_unit, "rp": rp, "scale": scale}
    print(f"  {cond}: {len(rm)} roles, scale={scale:.3f}")

# Pass 2: stream .pt files, project, store scalars
print("Pass 2: projecting per-question activations...")
result = {}
for cond in CONDITIONS:
    aa_unit = cond_axes[cond]["aa_unit"]
    rp = cond_axes[cond]["rp"]
    scale = cond_axes[cond]["scale"]
    cond_result = {}
    n_files = 0
    for activations_dir in get_activation_dirs(cond):
        for pt_file in sorted(activations_dir.glob("*.pt")):
            name = pt_file.stem
            group = classify(name)
            if group is None:
                continue
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            vecs = torch.stack(list(data.values())).squeeze(1).float()
            projs = ((vecs @ aa_unit) - rp) / scale
            cond_result[name] = {
                "group": group,
                "projs": projs.tolist(),
                "mean": float(projs.mean()),
            }
            n_files += 1
            del data, vecs, projs
    result[cond] = cond_result
    print(f"  {cond}: {n_files} roles projected")

out_path = REPO_ROOT / "results" / "character_training_projections.json"
with open(out_path, "w") as f:
    json.dump(result, f)
print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
