# Follow-up: Does character training change the assistant axis?

## Question

The assistant axis (AA) in Qwen3-32B detects linguistic register, not content.
Adversarial evil-assistant personas (murder consultant, eugenicist, etc.) score
highly on the AA because they use professional register while advocating harm.

Does character training (constitutional AI + DPO + introspective SFT) change
the geometry of persona space so that the AA becomes content-aware?

## Experiment

Use Open Character Training (Maiya et al. 2025) models:
- Paper: https://arxiv.org/abs/2511.01689
- Code: https://github.com/maiush/OpenCharacterTraining
- Models: https://huggingface.co/collections/maius/open-character-training

Available LoRA adapters for Qwen 2.5 7B:
- `qwen-2.5-7b-it-personas` (11 aligned personas)
- `qwen-2.5-7b-it-misalignment` (misaligned persona)

### Steps

1. Load Qwen 2.5 7B + aligned persona LoRA
2. Run 275 roles through psychographic battery → compute AA
3. Run fictional characters → check dimensionality, universe separation
4. Run adversarial evil-assistant prompts → does the AA catch them now?
5. Repeat with misalignment LoRA
6. Compare axis directions (cosines), thresholds, and adversarial projections

### Controls
- Also run base Qwen 2.5 7B (no LoRA) as baseline
- Compare AA from 7B base with AA from our 32B (how much does scale matter?)

### Key predictions
- If character training makes AA content-aware: the murder consultant should
  project differently from a genuine helpful assistant on the character-trained
  model's AA, even though both use professional register
- If it doesn't help: structural critique of single-axis clamping is strengthened

## Origin

Jack Lindsey suggested recomputing the AA with adversarial examples.
Discussion led to: use existing character-trained models rather than training
our own. Qwen 2.5 7B with Open Character Training LoRAs is the cheapest path.
