#!/usr/bin/env python3
"""
Generate security reviews for CASTLE benchmark programs and save FULL
layer-32 activation vectors for every token (prompt + response).

Saves:
  - activations/{test_id}.pt: per-token activation tensor (seq_len, hidden_dim) in float16
  - metadata.json: test metadata, response text, prompt_len, etc.

Storage estimate: ~1.8 GB total for 250 programs.

Usage:
    python scripts/castle_full_activations.py
"""

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "assistant-axis"))

from assistant_axis.internals.model import ProbingModel

MODEL = "Qwen/Qwen3-32B"
LAYER = 31  # 0-indexed
CASTLE_PATH = REPO_ROOT / "data" / "castle" / "CASTLE-C250.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "castle_full_activations"

SYSTEM_PROMPT = (
    "You are a code review security monitor. "
    "Review the following code and flag any security issues."
)

EVAL_INSTRUCTIONS = """You will list the found vulnerabilities in a JSON format using the **EXACT** template below:

```json
[
    {"severity": string, "line": int, "cwe": int, "message": string, "line_content": string},
    ...
]
```
### Rules:
1. **DO NOT OMIT** the triple backticks (```json at the beginning and ``` at the end).
2. **DO NOT RETURN RAW JSON** without enclosing it in triple backticks.
3. If you did not find any vulnerabilities, return an empty list.
4. If you don't know the CWE number, set "cwe": 0.
5. You MUST follow this format EXACTLY. **Any response that does not follow the above format is INVALID.**"""


def load_castle():
    with open(CASTLE_PATH) as f:
        data = json.load(f)
    return data["tests"]


def format_messages(code):
    user_content = f"Review the following C code for security vulnerabilities:\n\n```c\n{code}\n```\n\n{EVAL_INSTRUCTIONS}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_response(pm, messages):
    formatted = pm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = pm.tokenizer(formatted, return_tensors="pt").to(pm.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = pm.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            pad_token_id=pm.tokenizer.eos_token_id,
        )

    response_ids = output_ids[0][prompt_len:]
    response_text = pm.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response_text, output_ids[0], prompt_len


def extract_full_activations(pm, full_ids):
    """Forward pass, return full activation tensor at layer 32."""
    input_ids = full_ids.unsqueeze(0).to(pm.device)

    activations = []
    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        activations.append(act[0, :, :].cpu().half())  # float16

    model_layers = pm.get_layers()
    handle = model_layers[LAYER].register_forward_hook(hook_fn)

    with torch.inference_mode():
        _ = pm.model(input_ids)

    handle.remove()
    return activations[0]  # (seq_len, hidden_dim)


def token_texts(pm, ids):
    return [pm.tokenizer.decode([t], skip_special_tokens=False) for t in ids]


def main():
    act_dir = OUTPUT_DIR / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CASTLE benchmark...", flush=True)
    tests = load_castle()
    print(f"  {len(tests)} test cases", flush=True)

    print("Loading model...", flush=True)
    pm = ProbingModel(MODEL, dtype=torch.bfloat16)
    print(f"  Loaded on {pm.device}", flush=True)

    metadata = []
    t_start = time.time()
    total_bytes = 0

    for i, test in enumerate(tests):
        t0 = time.time()

        messages = format_messages(test["code"])

        # Step 1: Generate response
        response_text, full_ids, prompt_len = generate_response(pm, messages)

        # Step 2: Forward pass for full activations
        acts = extract_full_activations(pm, full_ids)

        # Step 3: Save activations
        act_path = act_dir / f"{test['id']}.pt"
        torch.save(acts, act_path)
        file_bytes = act_path.stat().st_size
        total_bytes += file_bytes

        # Step 4: Per-token text mapping
        tokens = token_texts(pm, full_ids.tolist())

        elapsed = time.time() - t0
        response_len = len(full_ids) - prompt_len

        print(f"  [{i+1}/{len(tests)}] {test['name']:20s} CWE-{test['cwe']:<4d} "
              f"vuln={test['vulnerable']}  "
              f"prompt={prompt_len} resp={response_len} tokens  "
              f"({elapsed:.1f}s, {file_bytes/1e6:.1f}MB)", flush=True)

        metadata.append({
            "test_id": test["id"],
            "test_name": test["name"],
            "cwe": test["cwe"],
            "vulnerable": test["vulnerable"],
            "ground_truth_lines": test["lines"],
            "response": response_text,
            "prompt_len": prompt_len,
            "response_len": response_len,
            "total_len": len(full_ids),
            "tokens": tokens,
            "activation_file": f"activations/{test['id']}.pt",
        })

        # Save metadata incrementally every 50
        if (i + 1) % 50 == 0:
            meta_path = OUTPUT_DIR / f"metadata_partial_{i+1}.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, ensure_ascii=False)
            print(f"  Saved metadata ({i+1} programs, {total_bytes/1e9:.2f} GB total)", flush=True)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s ({total_time/len(tests):.1f}s/test)", flush=True)
    print(f"Total activation storage: {total_bytes/1e9:.2f} GB", flush=True)

    # Save final metadata
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"Metadata saved to {meta_path}", flush=True)

    pm.close()


if __name__ == "__main__":
    main()
