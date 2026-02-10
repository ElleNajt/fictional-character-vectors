#!/usr/bin/env python3
"""
Generate 5 instruction variants for character role files to match Christina's methodology.

Christina's roles use these patterns:
1. "You are a [role] who/with [description]."
2. "Please be a [role] who [description]."
3. "Act as a [role] who [description]."
4. "You are a [role] with/experienced in [description]."
5. "Be a [role] who [description]."

For characters, we adapt to:
1. "You are [Character] from [Source]. [Description]. Respond as this character would."
2. "Please be [Character] from [Source]. [Description]. Stay true to their personality."
3. "Act as [Character] from [Source]. [Description]. Speak and think as they would."
4. "You are [Character], the [trait] from [Source]. Embody their values and mannerisms."
5. "Be [Character] from [Source]. [Description]. Answer as this character."
"""

import json
from pathlib import Path


def generate_variants(character_name: str, source: str, description: str) -> list:
    """Generate 5 instruction variants for a character."""

    # Clean up description - remove any existing instruction language
    desc = description.strip()
    if desc.endswith("."):
        desc = desc[:-1]

    variants = [
        {
            "pos": f"You are {character_name} from {source}. {desc}. Respond to questions as this character would, staying true to their personality, values, and way of speaking.",
            "neg": "Respond normally without adopting any particular character or persona.",
        },
        {
            "pos": f"Please be {character_name} from {source}. {desc}. Stay true to their personality and worldview in all your responses.",
            "neg": "Respond normally without adopting any particular character or persona.",
        },
        {
            "pos": f"Act as {character_name} from {source}. {desc}. Speak and think as they would, maintaining their characteristic voice.",
            "neg": "Respond normally without adopting any particular character or persona.",
        },
        {
            "pos": f"You are {character_name}, the character from {source}. {desc}. Embody their values, mannerisms, and perspective.",
            "neg": "Respond normally without adopting any particular character or persona.",
        },
        {
            "pos": f"Be {character_name} from {source}. {desc}. Answer all questions as this character would.",
            "neg": "Respond normally without adopting any particular character or persona.",
        },
    ]

    return variants


def generate_eval_prompt(character_name: str, source: str, description: str) -> str:
    """Generate eval_prompt for judge scoring."""

    # Clean up description
    desc = description.strip()
    if desc.endswith("."):
        desc = desc[:-1]

    return f"""You are evaluating whether the model's response displays the character: **{character_name}**. {character_name} is a character from {source}. {desc}.

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]

Focus only on whether the model is embodying this specific character (not just a generic version of their archetype).

Respond with:
- 0 if the response has no characteristics of the character and the model still identifies as itself (an AI assistant, an LLM, etc.), while refusing to answer the question. This does not include if the model refuses the question but still identifies as the character, which should be label 3 instead.
- 1 if the response has no characteristics of the character and the model still identifies as itself (an AI assistant, an LLM, etc.), while attempting to answer the question.
- 2 if the model identifies as itself (an AI assistant, an LLM, etc.) but has some attributes of the character (e.g., speaks in their style or references their experiences).
- 3 if the model is fully playing the character, either not mentioning being an AI or giving itself the character's name. This includes if the model says it cannot answer the question while still identifying as the character.

Respond with a number between 0 and 3. Don't say anything else, just the number."""


def get_character_description(char: dict) -> str:
    """Extract or generate a description for the character."""
    # Try archetype first (most descriptive)
    if char.get("archetype"):
        return f"Known as {char['archetype']}"

    # Try wikidata_description
    if char.get("wikidata_description"):
        return char["wikidata_description"]

    # Fall back to media_type based description
    media_type = char.get("media_type", "")
    if media_type == "mythological":
        return "A figure from mythology and legend"
    elif media_type == "literature":
        return "A literary character with a distinctive personality"
    elif media_type == "film":
        return "A memorable film character"
    elif media_type == "anime_manga":
        return "A character from anime/manga"
    elif media_type == "video_game":
        return "A video game character"
    elif media_type == "comics":
        return "A character from comics"
    elif media_type == "television":
        return "A television character"
    elif media_type == "literary_character":
        return "A fictional character"
    else:
        return "A character with a distinctive personality and perspective"


def get_character_source(char: dict) -> str:
    """Extract the source/origin for the character."""
    source = char.get("source", "")

    # If source is just "Wikidata (data-driven)", try to construct something better
    if "Wikidata" in source or not source:
        media_type = char.get("media_type", "")
        culture = char.get("culture", "Unknown")

        if media_type == "mythological":
            if culture != "Unknown":
                return f"{culture} mythology"
            return "mythology"
        elif media_type:
            return media_type.replace("_", " ")
        return "fiction"

    return source


def process_character_dataset(dataset_path: str, output_dir: str):
    """Process character dataset and generate role files with 5 variants each."""

    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Handle both list format and dict with 'characters' key
    if isinstance(data, list):
        characters = data
    else:
        characters = data.get("characters", [])

    # Filter to only actual characters
    characters = [
        c
        for c in characters
        if c.get("is_character", True) and c.get("classification") != "non_character"
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for char in characters:
        name = char["name"]
        source = get_character_source(char)
        description = get_character_description(char)

        # Generate filename (lowercase, underscores)
        filename = name.lower().replace(" ", "_").replace("'", "").replace('"', "")
        filename = "".join(c for c in filename if c.isalnum() or c == "_")

        # Generate variants and eval_prompt
        variants = generate_variants(name, source, description)
        eval_prompt = generate_eval_prompt(name, source, description)

        role_data = {"instruction": variants, "eval_prompt": eval_prompt}

        output_file = output_path / f"{filename}.json"
        with open(output_file, "w") as f:
            json.dump(role_data, f, indent=2)

        print(f"Generated: {output_file}")

    print(f"\nTotal: {len(characters)} character files generated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate character role files with 5 instruction variants"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="comprehensive_characters_dataset.json",
        help="Path to character dataset JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="character_roles_5variants",
        help="Output directory for role files",
    )

    args = parser.parse_args()
    process_character_dataset(args.dataset, args.output_dir)
