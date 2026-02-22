#!/usr/bin/env python3
"""
Split blogpost.org into main.org (core narrative) and appendix.org (supporting material).

Main narrative sections (in order):
  - Header (lines 1-5)
  - Introduction + Method + Prompting (lines 6-35)
  - Setup (lines 36-107)
  - Finding 1: Lu captures ~60% (lines 108-129)
  - Finding 2: Residual is universe separation (lines 130-210)
  - Finding 3: Generic vs distinctive (lines 211-285)
  - Directions vs Rankings (lines 683-784)
  - Logit lens summary (new, written inline)
  - Response diffing summary (new, written inline)
  - Steering experiment (lines 1832-1922)
  - Conclusion + Limitations (lines 1923-1946)

Appendix sections (in order):
  - What Lu's Roles Capture (lines 286-332)
  - Within-Universe Residuals (lines 333-682)
  - Beyond PC1: Later PCs (lines 785-1003)
  - Which Questions Differentiate Characters (lines 1004-1566)
  - Sample Responses (lines 1567-1766)
  - Question Sensitivity (lines 1767-1831)
  - Logit Lens full tables (lines 1947-2010)
  - Response Diffing full (lines 2011-2313)
  - Cross-Universe PC Alignment (lines 2314-end)
"""

from pathlib import Path


def read_lines(path):
    with open(path) as f:
        return f.readlines()


def extract(lines, start, end):
    """Extract lines[start-1:end] (1-indexed, inclusive)."""
    return lines[start - 1 : end]


def main():
    src = Path(__file__).parent.parent / "blogpost.org"
    lines = read_lines(src)
    total = len(lines)

    # --- MAIN ---
    main_parts = []

    # Header
    main_parts.extend(extract(lines, 1, 5))

    # Introduction + Method + Prompting
    main_parts.extend(extract(lines, 6, 35))

    # Setup
    main_parts.extend(extract(lines, 36, 107))

    # Finding 1
    main_parts.extend(extract(lines, 108, 129))

    # Finding 2
    main_parts.extend(extract(lines, 130, 210))

    # Finding 3
    main_parts.extend(extract(lines, 211, 285))

    # Directions vs Rankings (the key ambiguity)
    main_parts.extend(extract(lines, 683, 784))

    # Logit lens summary (bridging paragraph, not from original)
    main_parts.append("\n")
    main_parts.append("* What Do the Axes Mean? (Summary)\n")
    main_parts.append("\n")
    main_parts.append(
        "The logit lens---projecting PCA directions through the LM head to see which vocabulary tokens they point toward---reveals a striking asymmetry.\n"
    )
    main_parts.append("\n")
    main_parts.append(
        "Lu's role PCA directions decode to clean, interpretable token-level semantics: PC1 maps to register (colloquial vs. formal/archaic), PC3 to warmth vs. calculation, PC5 to childlike vs. hardened competence. The model's vocabulary is organized along the same axes that PCA finds in activation space.\n"
    )
    main_parts.append("\n")
    main_parts.append(
        "Residual PCA directions (after removing Lu's 275-role space) do /not/ yield interpretable logit lens results. The top tokens are code fragments, random subwords, and semantically incoherent text. Whatever the residual encodes, it doesn't reduce to vocabulary shifts---it likely lives in higher-order response structure.\n"
    )
    main_parts.append("\n")
    main_parts.append(
        "Full logit lens tables are in [[file:appendix.org::*Logit Lens: What Do the Axes Mean in Token Space?][the appendix]].\n"
    )
    main_parts.append("\n")

    # Response diffing summary (bridging paragraph)
    main_parts.append("* What Does the Residual Encode? (Summary)\n")
    main_parts.append("\n")
    main_parts.append(
        "Since the logit lens fails on residual directions, we compare the actual text produced by characters at opposite ends of each universe's residual PC1. The pattern is consistent across universes: residual PC1 separates /characters the model treats as specific people/---with personal experiences, relationships, and practical knowledge---from /characters it treats as archetypes, forces of nature, or philosophical authorities/.\n"
    )
    main_parts.append("\n")
    main_parts.append(
        "To validate this quantitatively, we run a held-out experiment: an LLM discovers distinguishing features from odd-ranked characters, then those features are coded on non-overlapping even-ranked characters and regressed against actual PC scores. The adjusted R^{2} is ~0.77 across universes (median), confirming the residual captures real, text-observable structure. Simple text statistics (no LLM) achieve comparable results (adj. R^{2} ~0.71).\n"
    )
    main_parts.append("\n")
    main_parts.append(
        "This is consistent with the parsimonious reading: Lu's roles capture the first-order vocabulary/register component, while the residual contains higher-order structure (anecdote vs. pronouncement, practical vs. philosophical framing) that doesn't reduce to individual token probabilities.\n"
    )
    main_parts.append("\n")
    main_parts.append(
        "Full response diffing analysis, qualitative examples, and feature regression details are in [[file:appendix.org::*Interpreting Residual PCs: Response Diffing][the appendix]].\n"
    )
    main_parts.append("\n")

    # Steering experiment
    main_parts.extend(extract(lines, 1832, 1922))

    # Conclusion + Limitations
    main_parts.extend(extract(lines, 1923, 1946))

    # --- APPENDIX ---
    appendix_parts = []
    appendix_parts.append("#+TITLE: Appendix: Stress-Testing Persona Space\n")
    appendix_parts.append("#+AUTHOR: Elle\n")
    appendix_parts.append("#+DATE: 2026-02-16\n")
    appendix_parts.append(
        '#+PROPERTY: header-args:python :results output drawer :python "./.venv/bin/python3" :async t :session blogpost :exports both\n'
    )
    appendix_parts.append("\n")

    # What Lu's Roles Capture
    appendix_parts.extend(extract(lines, 286, 332))

    # Within-Universe Residuals
    appendix_parts.extend(extract(lines, 333, 682))

    # Beyond PC1
    appendix_parts.extend(extract(lines, 785, 1003))

    # Which Questions Differentiate
    appendix_parts.extend(extract(lines, 1004, 1566))

    # Sample Responses
    appendix_parts.extend(extract(lines, 1567, 1766))

    # Question Sensitivity
    appendix_parts.extend(extract(lines, 1767, 1831))

    # Logit Lens full
    appendix_parts.extend(extract(lines, 1947, 2010))

    # Response Diffing full
    appendix_parts.extend(extract(lines, 2011, 2313))

    # Cross-Universe PC Alignment
    appendix_parts.extend(extract(lines, 2314, total))

    # Write files
    out_dir = Path(__file__).parent
    with open(out_dir / "main.org", "w") as f:
        f.writelines(main_parts)

    with open(out_dir / "appendix.org", "w") as f:
        f.writelines(appendix_parts)

    # Count lines
    main_count = len(main_parts)
    appendix_count = len(appendix_parts)
    print(f"main.org: {main_count} lines")
    print(f"appendix.org: {appendix_count} lines")
    print(f"Original: {total} lines")


if __name__ == "__main__":
    main()
