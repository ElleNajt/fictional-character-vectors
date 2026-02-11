#!/usr/bin/env python3
"""
Convert character JSON files to role instruction format for assistant-axis pipeline.

Reads from data/franchises/*.json and data/canonical_fictional_characters.json
Outputs to data/roles/instructions/*.json
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "roles" / "instructions"


def make_role_instructions(name: str, franchise: str) -> dict:
    """Create role instruction JSON in assistant-axis format.

    We intentionally omit descriptions to let the model's own representation
    of the character emerge, rather than biasing it with our descriptions.
    """
    return {
        "instruction": [
            {
                "pos": f"You are {name} from {franchise}. Respond as this character would."
            },
            {"pos": f"Embody {name} from {franchise}. Stay in character."},
            {"pos": f"You are roleplaying as {name} from {franchise}."},
            {"pos": f"Take on the persona of {name} from {franchise}."},
            {"pos": f"Act as {name} from {franchise}."},
        ]
    }


def safe_filename(name: str, franchise: str) -> str:
    """Create safe filename from character name and franchise."""

    def slugify(s: str) -> str:
        s = s.lower().replace(" ", "_").replace("'", "").replace("/", "_")
        return "".join(c for c in s if c.isalnum() or c == "_")

    franchise_slug = slugify(franchise)
    name_slug = slugify(name)
    return f"{franchise_slug}__{name_slug}"


def process_franchise_file(json_path: Path) -> int:
    """Process a franchise JSON file, return count of characters."""
    with open(json_path) as f:
        data = json.load(f)

    franchise = data["franchise"]
    count = 0

    for char in data["characters"]:
        name = char["name"]

        role_data = make_role_instructions(name, franchise)
        filename = safe_filename(name, franchise)

        output_path = OUTPUT_DIR / f"{filename}.json"
        with open(output_path, "w") as f:
            json.dump(role_data, f, indent=2)

        count += 1

    return count


def process_canonical_characters(json_path: Path) -> int:
    """Process canonical characters JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    # Handle dict with "characters" key or plain list
    if isinstance(data, dict):
        characters = data.get("characters", [])
    else:
        characters = data

    count = 0
    for char in characters:
        name = char["name"]
        source = char.get("source", "Unknown")

        role_data = make_role_instructions(name, source)
        filename = safe_filename(name, source)

        output_path = OUTPUT_DIR / f"{filename}.json"
        with open(output_path, "w") as f:
            json.dump(role_data, f, indent=2)

        count += 1

    return count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    # Process franchise files
    franchises_dir = DATA_DIR / "franchises"
    if franchises_dir.exists():
        for json_file in franchises_dir.glob("*.json"):
            count = process_franchise_file(json_file)
            print(f"  {json_file.name}: {count} characters")
            total += count

    # Process canonical characters
    canonical_path = DATA_DIR / "canonical_fictional_characters.json"
    if canonical_path.exists():
        count = process_canonical_characters(canonical_path)
        print(f"  canonical_fictional_characters.json: {count} characters")
        total += count

    print(f"\nTotal: {total} role files created in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
