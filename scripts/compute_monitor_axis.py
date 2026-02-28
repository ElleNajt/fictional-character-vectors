#!/usr/bin/env python3
"""
Compute monitor axis vectors from extracted activations.

For each monitor role (monitor, paranoid_monitor):
  axis = mean(monitor_activations) - mean(role_activations)

The axis points FROM role-playing TOWARD monitor behavior (analogous to
the assistant axis pointing from role-playing toward default assistant).

Usage:
    python scripts/compute_monitor_axis.py

Requires:
    - outputs/monitor_axis/activations/{monitor,paranoid_monitor}.pt
      (from slurm_monitor_axis.sh)
    - outputs/qwen3-32b_20260211_002840/vectors/*.pt
      (existing role vectors)
"""

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MONITOR_ACT_DIR = REPO_ROOT / "outputs" / "monitor_axis" / "activations"
ROLE_VECTORS_DIR = REPO_ROOT / "outputs" / "qwen3-32b_20260211_002840" / "vectors"
OUTPUT_DIR = REPO_ROOT / "data" / "role_vectors"
LAYER = 32


def load_monitor_mean(activation_file: Path) -> torch.Tensor:
    """Load monitor activations and compute mean vector at target layer."""
    data = torch.load(activation_file, map_location="cpu", weights_only=False)
    # Activations extracted with --layers 32: each value is (1, hidden_dim)
    # Activations extracted with all layers: each value is (n_layers, hidden_dim)
    acts = list(data.values())
    stacked = torch.stack(acts)  # (n_samples, n_layers, hidden_dim)

    if stacked.shape[1] == 1:
        # Single layer extracted (--layers 32)
        return stacked[:, 0, :].mean(dim=0)  # (hidden_dim,)
    else:
        # All layers extracted
        return stacked[:, LAYER, :].mean(dim=0)  # (hidden_dim,)


def load_role_mean() -> torch.Tensor:
    """Load existing role vectors and compute mean at target layer."""
    role_vectors = []
    for vec_file in sorted(ROLE_VECTORS_DIR.glob("*.pt")):
        data = torch.load(vec_file, map_location="cpu", weights_only=False)
        role = data.get("role", vec_file.stem)
        # Skip default roles (they're the assistant baseline, not role-playing)
        if "default" in role:
            continue
        # Skip monitor roles if they somehow ended up here
        if "monitor" in role:
            continue
        role_vectors.append(data["vector"][LAYER])  # (hidden_dim,)

    print(f"Loaded {len(role_vectors)} role vectors")
    return torch.stack(role_vectors).mean(dim=0)  # (hidden_dim,)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    role_mean = load_role_mean()

    monitor_roles = ["monitor", "paranoid_monitor"]
    for role_name in monitor_roles:
        act_file = MONITOR_ACT_DIR / f"{role_name}.pt"
        if not act_file.exists():
            print(f"Skipping {role_name}: {act_file} not found")
            continue

        monitor_mean = load_monitor_mean(act_file)
        axis = monitor_mean - role_mean

        output_file = OUTPUT_DIR / f"{role_name}_axis.pt"
        torch.save(axis, output_file)

        # Compare with assistant axis
        aa_file = OUTPUT_DIR / "assistant_axis.pt"
        if aa_file.exists():
            aa = torch.load(aa_file, map_location="cpu", weights_only=False)
            if aa.dim() > 1:
                aa = aa[LAYER]
            cosine = torch.nn.functional.cosine_similarity(
                axis.unsqueeze(0), aa.unsqueeze(0)
            ).item()
            print(f"{role_name}_axis: norm={axis.norm():.2f}, cosine(assistant_axis)={cosine:.3f}")
        else:
            print(f"{role_name}_axis: norm={axis.norm():.2f}")

        print(f"  Saved to {output_file}")


if __name__ == "__main__":
    main()
