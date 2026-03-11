"""
Extract per-token AA projections for adversarial clamping responses.

For each (character, condition, question), does a forward pass and saves
per-token projections onto the assistant axis at layer 50.

Output: adversarial_token_projections.json with per-token text + AA projection.
"""
import json
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "assistant-axis"))

from assistant_axis.internals.model import ProbingModel

MODEL = "Qwen/Qwen3-32B"
LAYER = 49  # 0-indexed, = layer 50

RESULTS_PATH = REPO_ROOT / "results" / "adversarial_clamping_comparison.json"
AA_PATH = REPO_ROOT / "data" / "role_vectors" / "assistant_axis.pt"
OUTPUT_PATH = REPO_ROOT / "results" / "adversarial_token_projections.json"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    aa_vec = torch.load(AA_PATH, weights_only=True)
    aa_l50 = aa_vec[LAYER].float()
    aa_l50 = aa_l50 / aa_l50.norm()

    print("Loading model...", flush=True)
    pm = ProbingModel(MODEL, dtype=torch.bfloat16)
    aa_l50 = aa_l50.to(pm.device)

    model_layers = pm.get_layers()
    results = []

    conditions = ["baseline", "aa_clamped", "villain_clamped"]

    for i, entry in enumerate(data):
        for condition in conditions:
            response_text = entry[condition]
            conversation = [
                {"role": "system", "content": entry["system_prompt"]},
                {"role": "user", "content": entry["question"]},
                {"role": "assistant", "content": response_text},
            ]

            formatted = pm.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            inputs = pm.tokenizer(formatted, return_tensors="pt").to(pm.device)
            input_ids = inputs.input_ids

            # Find where the assistant response starts
            # Tokenize without the assistant content to get prompt length
            prompt_conv = [
                {"role": "system", "content": entry["system_prompt"]},
                {"role": "user", "content": entry["question"]},
            ]
            prompt_formatted = pm.tokenizer.apply_chat_template(
                prompt_conv,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = pm.tokenizer(prompt_formatted, return_tensors="pt")
            prompt_len = prompt_ids.input_ids.shape[1]

            # Forward pass with hook
            activations = []
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                activations.append(act[0].float().cpu())

            handle = model_layers[LAYER].register_forward_hook(hook_fn)
            with torch.inference_mode():
                _ = pm.model(**inputs)
            handle.remove()

            acts = activations[0]  # (seq_len, hidden_dim)
            # Project response tokens onto AA
            response_acts = acts[prompt_len:]
            projs = (response_acts @ aa_l50.cpu()).tolist()

            # Get per-token text for response portion
            response_ids = input_ids[0][prompt_len:].tolist()
            tokens = [pm.tokenizer.decode([t], skip_special_tokens=False) for t in response_ids]

            results.append({
                "character": entry["character"],
                "condition": condition,
                "question": entry["question"],
                "tokens": tokens,
                "projections": projs,
                "mean_proj": sum(projs) / len(projs) if projs else 0,
            })

            print(f"  [{i+1}/{len(data)}] {entry['character']:30s} {condition:20s} "
                  f"{len(projs):4d} tokens  mean={results[-1]['mean_proj']:+.1f}", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"\nSaved {len(results)} entries to {OUTPUT_PATH}", flush=True)

    pm.close()


if __name__ == "__main__":
    main()
