"""Merge a LoRA adapter into the base model and save to disk."""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_adapter", required=True)
    parser.add_argument("--subfolder", default=None, help="Subfolder within adapter repo (e.g. 'goodness')")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and (output_dir / "config.json").exists():
        print(f"Merged model already exists at {output_dir}, skipping")
        return

    print(f"Loading base model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    lora_kwargs = {}
    if args.subfolder:
        lora_kwargs["subfolder"] = args.subfolder
    print(f"Loading and merging LoRA: {args.lora_adapter} (subfolder={args.subfolder})")
    model = PeftModel.from_pretrained(model, args.lora_adapter, **lora_kwargs)
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done")


if __name__ == "__main__":
    main()
