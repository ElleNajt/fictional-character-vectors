#!/usr/bin/env python3
"""
Fetch fictional characters from Wikidata ranked by cultural importance.

Uses Wikipedia language count as primary metric of cultural spread.
Also fetches celebrities/historical figures for comparison.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests


class WikidataCharacterFetcher:
    """Fetch and rank fictional characters from Wikidata."""

    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

    # Character type QIDs
    CHARACTER_TYPES = {
        "fictional_character": "Q95074",
        "fictional_human": "Q15632617",
        "literary_character": "Q15773347",
        "film_character": "Q15773317",
        "television_character": "Q15773347",
        "anime_character": "Q15711870",
        "video_game_character": "Q15978536",
        "comics_character": "Q1114461",
        "mythological_character": "Q4271324",
    }

    def __init__(self, user_agent: str = "FictionalCharacterResearch/1.0"):
        """Initialize fetcher with user agent."""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def query_sparql(self, query: str) -> Optional[Dict]:
        """Execute SPARQL query against Wikidata endpoint."""
        try:
            response = self.session.get(
                self.SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying Wikidata: {e}")
            return None

    def get_fictional_characters(self, limit: int = 1000) -> List[Dict]:
        """
        Get fictional characters ranked by Wikipedia language count.

        This queries for any entity that is an instance of (or subclass of)
        a fictional character type, and counts how many Wikipedia language
        editions have articles about them.
        """
        query = f"""
        SELECT ?character ?characterLabel ?characterDescription 
               (COUNT(DISTINCT ?article) as ?languageCount)
               ?birthplace ?birthplaceLabel
               ?gender ?genderLabel
               ?occupation ?occupationLabel
        WHERE {{
          # Get characters - instance of any fictional character type
          VALUES ?characterType {{ 
            wd:Q95074          # character
            wd:Q15632617       # fictional human
            wd:Q15773347       # literary character
            wd:Q4271324        # mythological character
          }}
          ?character (wdt:P31/wdt:P279*) ?characterType .
          
          # Count Wikipedia articles across languages
          ?article schema:about ?character .
          ?article schema:isPartOf [ wikibase:wikiGroup "wikipedia" ] .
          
          # Optional properties
          OPTIONAL {{ ?character wdt:P19 ?birthplace }}
          OPTIONAL {{ ?character wdt:P21 ?gender }}
          OPTIONAL {{ ?character wdt:P106 ?occupation }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        GROUP BY ?character ?characterLabel ?characterDescription 
                 ?birthplace ?birthplaceLabel ?gender ?genderLabel 
                 ?occupation ?occupationLabel
        HAVING (?languageCount > 10)  # Filter to characters in 10+ languages
        ORDER BY DESC(?languageCount)
        LIMIT {limit}
        """

        print(f"Querying Wikidata for top {limit} fictional characters...")
        result = self.query_sparql(query)

        if not result:
            return []

        characters = []
        for binding in result.get("results", {}).get("bindings", []):
            character = {
                "name": binding["characterLabel"]["value"],
                "description": binding.get("characterDescription", {}).get("value", ""),
                "wikidata_id": binding["character"]["value"].split("/")[-1],
                "wikipedia_languages": int(binding["languageCount"]["value"]),
                "type": "fictional_character",
                "gender": binding.get("genderLabel", {}).get("value", ""),
                "occupation": binding.get("occupationLabel", {}).get("value", ""),
            }
            characters.append(character)

        print(f"Found {len(characters)} fictional characters")
        return characters

    def get_specific_character_type(
        self, character_type_qid: str, limit: int = 500
    ) -> List[Dict]:
        """Get characters of a specific type (e.g., anime, video game)."""
        query = f"""
        SELECT ?character ?characterLabel ?characterDescription 
               (COUNT(DISTINCT ?article) as ?languageCount)
        WHERE {{
          ?character (wdt:P31/wdt:P279*) wd:{character_type_qid} .
          ?article schema:about ?character .
          ?article schema:isPartOf [ wikibase:wikiGroup "wikipedia" ] .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        GROUP BY ?character ?characterLabel ?characterDescription
        HAVING (?languageCount > 5)
        ORDER BY DESC(?languageCount)
        LIMIT {limit}
        """

        result = self.query_sparql(query)
        if not result:
            return []

        characters = []
        for binding in result.get("results", {}).get("bindings", []):
            character = {
                "name": binding["characterLabel"]["value"],
                "description": binding.get("characterDescription", {}).get("value", ""),
                "wikidata_id": binding["character"]["value"].split("/")[-1],
                "wikipedia_languages": int(binding["languageCount"]["value"]),
                "type": character_type_qid,
            }
            characters.append(character)

        return characters

    def get_celebrities(self, limit: int = 500) -> List[Dict]:
        """
        Get real celebrities/historical figures for comparison.

        Useful for testing if fictional characters and real people occupy
        the same persona space.
        """
        query = f"""
        SELECT ?person ?personLabel ?personDescription 
               (COUNT(DISTINCT ?article) as ?languageCount)
               ?birthYear ?deathYear
               ?occupation ?occupationLabel
        WHERE {{
          # Get humans (real people)
          ?person wdt:P31 wd:Q5 .  # instance of human
          
          # Count Wikipedia articles
          ?article schema:about ?person .
          ?article schema:isPartOf [ wikibase:wikiGroup "wikipedia" ] .
          
          # Get birth/death years
          OPTIONAL {{ 
            ?person wdt:P569 ?birthDate .
            BIND(YEAR(?birthDate) as ?birthYear)
          }}
          OPTIONAL {{ 
            ?person wdt:P570 ?deathDate .
            BIND(YEAR(?deathDate) as ?deathYear)
          }}
          
          # Get occupation
          OPTIONAL {{ ?person wdt:P106 ?occupation }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        GROUP BY ?person ?personLabel ?personDescription 
                 ?birthYear ?deathYear ?occupation ?occupationLabel
        HAVING (?languageCount > 50)  # Only very notable people
        ORDER BY DESC(?languageCount)
        LIMIT {limit}
        """

        print(f"Querying Wikidata for top {limit} celebrities/historical figures...")
        result = self.query_sparql(query)

        if not result:
            return []

        people = []
        for binding in result.get("results", {}).get("bindings", []):
            person = {
                "name": binding["personLabel"]["value"],
                "description": binding.get("personDescription", {}).get("value", ""),
                "wikidata_id": binding["person"]["value"].split("/")[-1],
                "wikipedia_languages": int(binding["languageCount"]["value"]),
                "type": "real_person",
                "birth_year": binding.get("birthYear", {}).get("value", ""),
                "death_year": binding.get("deathYear", {}).get("value", ""),
                "occupation": binding.get("occupationLabel", {}).get("value", ""),
            }
            people.append(person)

        print(f"Found {len(people)} celebrities/historical figures")
        return people

    def get_mythological_figures(self, limit: int = 300) -> List[Dict]:
        """Get mythological figures (gods, heroes, etc.)."""
        query = f"""
        SELECT ?character ?characterLabel ?characterDescription 
               (COUNT(DISTINCT ?article) as ?languageCount)
               ?mythology ?mythologyLabel
        WHERE {{
          VALUES ?mythType {{ 
            wd:Q4271324    # mythological character
            wd:Q178561     # deity
            wd:Q13002315   # mythical character
          }}
          ?character (wdt:P31/wdt:P279*) ?mythType .
          
          ?article schema:about ?character .
          ?article schema:isPartOf [ wikibase:wikiGroup "wikipedia" ] .
          
          OPTIONAL {{ ?character wdt:P361 ?mythology }}  # part of mythology
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        GROUP BY ?character ?characterLabel ?characterDescription 
                 ?mythology ?mythologyLabel
        HAVING (?languageCount > 10)
        ORDER BY DESC(?languageCount)
        LIMIT {limit}
        """

        print(f"Querying Wikidata for mythological figures...")
        result = self.query_sparql(query)

        if not result:
            return []

        characters = []
        for binding in result.get("results", {}).get("bindings", []):
            character = {
                "name": binding["characterLabel"]["value"],
                "description": binding.get("characterDescription", {}).get("value", ""),
                "wikidata_id": binding["character"]["value"].split("/")[-1],
                "wikipedia_languages": int(binding["languageCount"]["value"]),
                "type": "mythological",
                "mythology": binding.get("mythologyLabel", {}).get("value", ""),
            }
            characters.append(character)

        print(f"Found {len(characters)} mythological figures")
        return characters


def deduplicate_characters(characters: List[Dict]) -> List[Dict]:
    """Remove duplicates based on Wikidata ID."""
    seen_ids = set()
    unique_chars = []

    for char in characters:
        qid = char["wikidata_id"]
        if qid not in seen_ids:
            seen_ids.add(qid)
            unique_chars.append(char)

    return unique_chars


def main():
    """Main execution function."""
    print("=" * 80)
    print("Wikidata Fictional Character Fetcher")
    print("=" * 80)

    fetcher = WikidataCharacterFetcher()

    all_characters = []

    # 1. Get general fictional characters (top 1000)
    print("\n1. Fetching general fictional characters...")
    fictional = fetcher.get_fictional_characters(limit=1000)
    all_characters.extend(fictional)
    time.sleep(2)  # Be nice to Wikidata servers

    # 2. Get mythological figures
    print("\n2. Fetching mythological figures...")
    mythological = fetcher.get_mythological_figures(limit=300)
    all_characters.extend(mythological)
    time.sleep(2)

    # 3. Get celebrities for comparison
    print("\n3. Fetching celebrities/historical figures...")
    celebrities = fetcher.get_celebrities(limit=500)
    time.sleep(2)

    # Deduplicate
    print("\n4. Deduplicating characters...")
    all_characters = deduplicate_characters(all_characters)
    celebrities = deduplicate_characters(celebrities)

    # Sort by language count
    all_characters.sort(key=lambda x: x["wikipedia_languages"], reverse=True)
    celebrities.sort(key=lambda x: x["wikipedia_languages"], reverse=True)

    # Create output
    output = {
        "metadata": {
            "generated_date": datetime.now().isoformat(),
            "source": "Wikidata SPARQL queries",
            "total_fictional_characters": len(all_characters),
            "total_real_people": len(celebrities),
            "ranking_metric": "Wikipedia language count",
        },
        "fictional_characters": all_characters,
        "real_people": celebrities,
    }

    # Save to JSON
    output_file = "wikidata_characters_ranked.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"Successfully saved {len(all_characters)} fictional characters")
    print(f"and {len(celebrities)} real people to:")
    print(f"  {output_file}")
    print(f"{'=' * 80}")

    # Print top 20 for preview
    print("\nTop 20 Fictional Characters by Wikipedia Languages:")
    print("-" * 80)
    for i, char in enumerate(all_characters[:20], 1):
        print(
            f"{i:2d}. {char['name']:40s} ({char['wikipedia_languages']:3d} languages)"
        )

    print("\nTop 20 Real People for Comparison:")
    print("-" * 80)
    for i, person in enumerate(celebrities[:20], 1):
        print(
            f"{i:2d}. {person['name']:40s} ({person['wikipedia_languages']:3d} languages)"
        )

    return output


if __name__ == "__main__":
    main()
