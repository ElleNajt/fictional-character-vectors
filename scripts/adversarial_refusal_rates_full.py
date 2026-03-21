"""
Re-run ALL adversarial cases N times to measure refusal rates.

Reads (character, question, system_prompt) from the existing comparison JSON
and generates N responses per condition (baseline + aa_clamped).

Usage:
    srun --job-name=refusal_full --gres=gpu:4 --mem=200G \
      python3 scripts/adversarial_refusal_rates_full.py [--n 10]

Output: results/adversarial_refusal_rates_full.json
"""
import torch
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistant_axis import ActivationSteering, generate_response
from huggingface_hub import hf_hub_download
from assistant_axis import load_capping_config, build_capping_steerer

MODEL_NAME = "Qwen/Qwen3-32B"
REPO_ROOT = Path('/workspace-vast/lnajt')
RESULTS_PATH = REPO_ROOT / 'results' / 'adversarial_clamping_comparison.json'
OUTPUT_PATH = REPO_ROOT / 'results' / 'adversarial_refusal_rates_full.json'

SKIP_CHARS = {"Forseti_Heroic", "Samwise_Heroic", "Voldemort_Gloating"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Generations per (character, question, condition)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load all (character, question, system_prompt) from existing results
    with open(RESULTS_PATH) as f:
        existing = json.load(f)

    pairs = []
    seen = set()
    for e in existing:
        if e['character'] in SKIP_CHARS:
            continue
        key = (e['character'], e['question'])
        if key not in seen:
            seen.add(key)
            pairs.append((e['character'], e['question'], e['system_prompt']))

    print(f"Total (character, question) pairs: {len(pairs)}")

    # Resume support
    results = []
    done_counts = {}
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
        for r in results:
            key = (r['character'], r['question'], r['condition'])
            done_counts[key] = done_counts.get(key, 0) + 1
        print(f"Resuming: {len(results)} responses already generated")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Build AA steerer
    capping_config_path = hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        filename="qwen-3-32b/capping_config.pt", repo_type="dataset")
    capping_config = load_capping_config(capping_config_path)
    aa_steerer = build_capping_steerer(model, capping_config, 'layers_46:54-p0.25')

    total_gens = len(pairs) * 2 * args.n
    done = len(results)

    for pair_idx, (char, question, system_prompt) in enumerate(pairs):
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        for condition in ["baseline", "aa_clamped"]:
            already = done_counts.get((char, question, condition), 0)
            remaining = args.n - already

            for i in range(remaining):
                if condition == "baseline":
                    response = generate_response(model, tokenizer, conversation, max_new_tokens=512)
                else:
                    with aa_steerer:
                        response = generate_response(model, tokenizer, conversation, max_new_tokens=512)

                results.append({
                    'character': char,
                    'question': question,
                    'condition': condition,
                    'run': already + i,
                    'response': response,
                })
                done += 1
                if done % 10 == 0:
                    print(f"[{done}/{total_gens}] {char:30s} {condition:12s} run={already+i}")

                # Save periodically
                if done % 100 == 0:
                    with open(OUTPUT_PATH, 'w') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} responses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
