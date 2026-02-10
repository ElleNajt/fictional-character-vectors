#!/usr/bin/env python3
"""
Convert canonical_fictional_characters.json to role files for the assistant-axis pipeline.

Usage:
    python convert_characters_to_roles.py \
        --characters data/canonical_fictional_characters.json \
        --output_dir outputs/character_roles \
        --max_characters 20
"""

import argparse
import json
import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Convert character name to safe filename."""
    # Replace spaces and special chars with underscores
    safe = re.sub(r"[^\w\s-]", "", name)
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe.lower().strip("_")


def create_role_instruction(character: dict) -> dict:
    """Create role instruction for a character."""
    name = character["name"]
    source = character.get("source", "")
    archetype = character.get("archetype", "")
    notes = character.get("notes", "")

    # Create instruction text
    instruction_text = f"You are {name}"
    if source:
        instruction_text += f" from {source}"
    instruction_text += "."

    if archetype:
        instruction_text += f" You are known as {archetype}."

    if notes:
        instruction_text += f" {notes}"

    instruction_text += " Respond to questions as this character would, staying true to their personality, values, and way of speaking."

    # Create role file format (simplified - just one instruction)
    return {
        "instruction": [
            {
                "pos": instruction_text,
                "neg": "Respond normally without adopting any particular character or persona.",
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert characters to role files")
    parser.add_argument(
        "--characters",
        required=True,
        help="Path to canonical_fictional_characters.json",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for role files"
    )
    parser.add_argument(
        "--max_characters",
        type=int,
        default=20,
        help="Maximum number of characters to convert",
    )
    args = parser.parse_args()

    # Load characters
    with open(args.characters) as f:
        data = json.load(f)

    # Filter to only actual characters (exclude historical events, etc.)
    all_characters = [
        e for e in data["characters"] if e.get("classification") == "character"
    ]
    characters = all_characters[: args.max_characters]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(characters)} characters to role files...")

    for character in characters:
        name = character["name"]
        safe_name = sanitize_filename(name)

        role_data = create_role_instruction(character)

        output_file = output_dir / f"{safe_name}.json"
        with open(output_file, "w") as f:
            json.dump(role_data, f, indent=2)

        print(f"  Created: {output_file.name}")

    print(f"\nDone! Created {len(characters)} role files in {output_dir}")


if __name__ == "__main__":
    main()
