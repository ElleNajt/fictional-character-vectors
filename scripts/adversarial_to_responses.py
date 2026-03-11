"""
Convert adversarial_clamping_comparison.json into response JSONL files
matching the format expected by pipeline/2_activations.py.

Creates one JSONL file per (character, condition) pair, e.g.:
  adversarial_baseline__HAL_direct.jsonl
  adversarial_aa_clamped__HAL_direct.jsonl
  adversarial_villain_clamped__HAL_direct.jsonl
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "adversarial_clamping_comparison.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "adversarial_clamping" / "responses"

CONDITIONS = ["baseline", "aa_clamped", "villain_clamped"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    # Group by character
    by_char = defaultdict(list)
    for entry in data:
        by_char[entry["character"]].append(entry)

    for char_name, entries in by_char.items():
        for condition in CONDITIONS:
            filename = f"adversarial_{condition}__{char_name}.jsonl"
            out_path = OUTPUT_DIR / filename
            with open(out_path, "w") as f:
                for i, entry in enumerate(entries):
                    record = {
                        "system_prompt": entry["system_prompt"],
                        "prompt_index": 0,
                        "question_index": i,
                        "question": entry["question"],
                        "conversation": [
                            {"role": "system", "content": entry["system_prompt"]},
                            {"role": "user", "content": entry["question"]},
                            {"role": "assistant", "content": entry[condition]},
                        ],
                        "label": condition,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Wrote {len(entries)} entries to {filename}")

    total = len(by_char) * len(CONDITIONS)
    print(f"\nDone: {total} JSONL files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
