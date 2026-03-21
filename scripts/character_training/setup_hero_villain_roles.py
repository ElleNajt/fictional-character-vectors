#!/usr/bin/env python3
"""Symlink hero/villain role files from main data directory into character training data dirs."""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
LABELS = REPO_ROOT / "data" / "hero_villain_labels.json"
SOURCE_ROLES = REPO_ROOT / "data" / "roles" / "instructions"

HERO_DIR = SCRIPT_DIR / "data" / "hero_roles"
VILLAIN_DIR = SCRIPT_DIR / "data" / "villain_roles"

with open(LABELS) as f:
    labels = json.load(f)

for group, target_dir in [("hero", HERO_DIR), ("villain", VILLAIN_DIR)]:
    target_dir.mkdir(parents=True, exist_ok=True)
    # Clear old symlinks
    for existing in target_dir.glob("*.json"):
        if existing.is_symlink():
            existing.unlink()
    linked = 0
    missing = []
    for name in labels[group]:
        src = SOURCE_ROLES / f"{name}.json"
        if src.exists():
            dst = target_dir / f"{name}.json"
            dst.symlink_to(src)
            linked += 1
        else:
            missing.append(name)
    print(f"{group}: linked {linked}/{len(labels[group])}")
    if missing:
        print(f"  missing: {missing}")
