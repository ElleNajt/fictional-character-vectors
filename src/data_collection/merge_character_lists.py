#!/usr/bin/env python3
"""
Merge hand-curated character list with Wikidata results.

Combines the comprehensive manual list with data-driven Wikidata rankings,
ensuring cultural diversity and archetypical representation.
"""

import json
from collections import Counter
from typing import Dict, List, Set


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, filepath: str):
    """Save to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_name(name: str) -> str:
    """Normalize character name for matching."""
    # Remove common suffixes and variations
    name = name.lower().strip()
    # Remove parentheticals like " (character)"
    if "(" in name:
        name = name.split("(")[0].strip()
    # Remove possessive 's
    name = name.replace("'s", "")
    return name


def find_matches(manual_chars: List[Dict], wikidata_chars: List[Dict]) -> Dict:
    """
    Find characters that appear in both lists.

    Returns dict with:
      - matched: list of (manual_char, wikidata_char) tuples
      - manual_only: characters only in manual list
      - wikidata_only: characters only in wikidata list
    """
    # Create normalized name index for wikidata
    wikidata_by_name = {}
    for char in wikidata_chars:
        norm_name = normalize_name(char["name"])
        wikidata_by_name[norm_name] = char

    matched = []
    manual_only = []

    for manual_char in manual_chars:
        norm_name = normalize_name(manual_char["name"])

        if norm_name in wikidata_by_name:
            matched.append((manual_char, wikidata_by_name[norm_name]))
            del wikidata_by_name[norm_name]  # Remove from dict
        else:
            manual_only.append(manual_char)

    wikidata_only = list(wikidata_by_name.values())

    return {
        "matched": matched,
        "manual_only": manual_only,
        "wikidata_only": wikidata_only,
    }


def merge_character_data(manual_char: Dict, wikidata_char: Dict) -> Dict:
    """Merge data from both sources, prioritizing manual curation."""
    merged = manual_char.copy()

    # Add Wikidata fields
    merged["wikidata_id"] = wikidata_char.get("wikidata_id", "")
    merged["wikipedia_languages"] = wikidata_char.get(
        "sitelinks", wikidata_char.get("wikipedia_languages", 0)
    )
    merged["wikidata_description"] = wikidata_char.get("description", "")

    return merged


def analyze_cultural_coverage(characters: List[Dict]) -> Dict:
    """Analyze cultural diversity of character list."""
    cultures = Counter(char.get("culture", "Unknown") for char in characters)
    media_types = Counter(char.get("media_type", "Unknown") for char in characters)
    eras = Counter(char.get("era", "Unknown") for char in characters)

    return {
        "total_characters": len(characters),
        "cultures": dict(cultures.most_common()),
        "media_types": dict(media_types.most_common()),
        "eras": dict(eras.most_common()),
        "unique_cultures": len(cultures),
        "unique_media_types": len(media_types),
    }


def create_hybrid_dataset(
    manual_file: str, wikidata_file: str, top_n_wikidata: int = 200
) -> Dict:
    """
    Create hybrid dataset combining manual curation with Wikidata rankings.

    Strategy:
    1. Start with all manually curated characters
    2. Merge with Wikidata where matches exist (add language count)
    3. Add top N characters from Wikidata that aren't in manual list
    4. Ensure cultural diversity benchmarks are met
    """
    print("Loading datasets...")
    manual_data = load_json(manual_file)
    wikidata_data = load_json(wikidata_file)

    manual_chars = manual_data["characters"]
    wikidata_chars = wikidata_data["fictional_characters"]

    print(f"Manual list: {len(manual_chars)} characters")
    print(f"Wikidata list: {len(wikidata_chars)} characters")

    # Find matches
    print("\nFinding matches between lists...")
    matches = find_matches(manual_chars, wikidata_chars)

    print(f"  Matched: {len(matches['matched'])}")
    print(f"  Manual only: {len(matches['manual_only'])}")
    print(f"  Wikidata only: {len(matches['wikidata_only'])}")

    # Create merged list
    print("\nMerging character data...")
    merged_characters = []

    # Add all matched characters (with enhanced data)
    for manual_char, wikidata_char in matches["matched"]:
        merged = merge_character_data(manual_char, wikidata_char)
        merged_characters.append(merged)

    # Add manual-only characters (keep all manual curation)
    for char in matches["manual_only"]:
        char_copy = char.copy()
        char_copy["wikipedia_languages"] = 0  # Not found in Wikidata
        merged_characters.append(char_copy)

    # Add top N from Wikidata that aren't in manual list
    print(f"\nAdding top {top_n_wikidata} characters from Wikidata...")
    wikidata_additions = []
    for char in matches["wikidata_only"][:top_n_wikidata]:
        # Convert Wikidata format to manual format
        new_char = {
            "name": char["name"],
            "source": "Wikidata (data-driven)",
            "media_type": char.get("type", "unknown"),
            "culture": "Unknown",  # Would need additional queries
            "era": "Unknown",
            "archetype": char.get("description", ""),
            "wikidata_id": char["wikidata_id"],
            "wikipedia_languages": char.get(
                "sitelinks", char.get("wikipedia_languages", 0)
            ),
            "notes": "Added from Wikidata top rankings",
        }
        wikidata_additions.append(new_char)

    merged_characters.extend(wikidata_additions)

    # Sort by Wikipedia language count (where available)
    merged_characters.sort(key=lambda x: x.get("wikipedia_languages", 0), reverse=True)

    # Analyze coverage
    print("\nAnalyzing cultural coverage...")
    coverage = analyze_cultural_coverage(merged_characters)

    # Create output
    output = {
        "metadata": {
            "title": "Comprehensive Fictional Characters Dataset (Hybrid)",
            "description": "Combination of hand-curated list and data-driven Wikidata rankings",
            "total_characters": len(merged_characters),
            "sources": [
                "Manual curation (cultural diversity ensured)",
                "Wikidata SPARQL queries (Wikipedia language count)",
            ],
            "manual_characters": len(manual_chars),
            "wikidata_additions": len(wikidata_additions),
            "matched_characters": len(matches["matched"]),
        },
        "cultural_coverage": coverage,
        "characters": merged_characters,
        "real_people_for_comparison": wikidata_data.get("real_people", [])[:200],
    }

    return output


def main():
    """Main execution."""
    print("=" * 80)
    print("Merging Manual and Wikidata Character Lists")
    print("=" * 80)

    # File paths (relative to working directory)
    manual_file = "data/canonical_fictional_characters.json"
    wikidata_file = "wikidata_characters_ranked.json"
    output_file = "comprehensive_characters_dataset.json"

    # Create hybrid dataset
    hybrid_data = create_hybrid_dataset(
        manual_file,
        wikidata_file,
        top_n_wikidata=200,  # Add top 200 from Wikidata
    )

    # Save
    save_json(hybrid_data, output_file)

    print(f"\n{'=' * 80}")
    print(f"Hybrid dataset created successfully!")
    print(f"Total characters: {hybrid_data['metadata']['total_characters']}")
    print(f"Saved to: {output_file}")
    print(f"{'=' * 80}")

    # Print coverage stats
    print("\nCultural Coverage:")
    for culture, count in list(hybrid_data["cultural_coverage"]["cultures"].items())[
        :10
    ]:
        print(f"  {culture:30s}: {count:4d}")

    print(f"\nTop 20 Characters by Wikipedia Languages:")
    for i, char in enumerate(hybrid_data["characters"][:20], 1):
        langs = char.get("wikipedia_languages", 0)
        print(f"  {i:2d}. {char['name']:40s} ({langs:3d} languages)")


if __name__ == "__main__":
    main()
