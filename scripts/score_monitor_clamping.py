#!/usr/bin/env python3
"""
Score monitor clamping experiment results against CASTLE ground truth.

Parses model JSON responses and computes:
  - True positive rate (vulnerability detected in vulnerable code)
  - False positive rate (vulnerability reported in clean code)
  - CWE accuracy (correct CWE identified)
  - Line accuracy (correct vulnerable line identified)
  - Precision, recall, F1

Usage:
    python scripts/score_monitor_clamping.py
    python scripts/score_monitor_clamping.py --results outputs/monitor_clamping/results_baseline.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "outputs" / "monitor_clamping"


def parse_json_response(response):
    """Extract JSON vulnerability list from model response.

    The model should return a fenced JSON block like:
    ```json
    [{"severity": ..., "line": ..., "cwe": ..., "message": ..., "line_content": ...}]
    ```

    Returns parsed list on success, None if response is unparseable.
    """
    # Try to find fenced JSON block
    match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Try to find any JSON array with objects
    match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    # Try to find empty array (no vulnerabilities found)
    if re.search(r'\[\s*\]', response):
        return []

    return None


def score_result(result):
    """Score a single result against ground truth.

    Returns dict with scoring metrics.
    """
    parse_error = None
    parsed = None
    try:
        parsed = parse_json_response(result["response"])
    except json.JSONDecodeError as e:
        parse_error = str(e)

    score = {
        "test_id": result["test_id"],
        "test_name": result["test_name"],
        "condition": result["condition"],
        "ground_truth_vulnerable": result["vulnerable"],
        "ground_truth_cwe": result["cwe"],
        "ground_truth_lines": result["ground_truth_lines"],
        "parse_success": parsed is not None,
        "parse_error": parse_error,
        "num_findings": 0,
        "detected_vulnerable": False,
        "correct_cwe": False,
        "correct_line": False,
        "reported_cwes": [],
        "reported_lines": [],
    }

    if parsed is None:
        return score

    score["num_findings"] = len(parsed)
    score["detected_vulnerable"] = len(parsed) > 0

    reported_cwes = set()
    reported_lines = set()
    for finding in parsed:
        if isinstance(finding, dict):
            cwe = finding.get("cwe", 0)
            if cwe:
                reported_cwes.add(int(cwe))
            line = finding.get("line", 0)
            if line:
                reported_lines.add(int(line))

    score["reported_cwes"] = sorted(reported_cwes)
    score["reported_lines"] = sorted(reported_lines)

    if result["vulnerable"]:
        score["correct_cwe"] = result["cwe"] in reported_cwes
        score["correct_line"] = bool(
            set(result["ground_truth_lines"]) & reported_lines
        )

    return score


def compute_metrics(scores):
    """Compute aggregate metrics from scored results."""
    vulnerable = [s for s in scores if s["ground_truth_vulnerable"]]
    clean = [s for s in scores if not s["ground_truth_vulnerable"]]

    # Parse success rate
    parse_ok = sum(1 for s in scores if s["parse_success"])

    # True positive: vulnerable code where we detected something
    tp = sum(1 for s in vulnerable if s["detected_vulnerable"])
    fn = sum(1 for s in vulnerable if not s["detected_vulnerable"])

    # False positive: clean code where we detected something
    fp = sum(1 for s in clean if s["detected_vulnerable"])
    tn = sum(1 for s in clean if not s["detected_vulnerable"])

    # CWE accuracy (among true positives)
    cwe_correct = sum(1 for s in vulnerable if s["correct_cwe"])
    line_correct = sum(1 for s in vulnerable if s["correct_line"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "n_total": len(scores),
        "n_vulnerable": len(vulnerable),
        "n_clean": len(clean),
        "parse_success": f"{parse_ok}/{len(scores)}",
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cwe_accuracy": f"{cwe_correct}/{len(vulnerable)}",
        "line_accuracy": f"{line_correct}/{len(vulnerable)}",
        "tpr": f"{tp}/{len(vulnerable)}",
        "fpr": f"{fp}/{len(clean)}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                        help="Path to results JSON (default: results_all.json)")
    args = parser.parse_args()

    if args.results:
        results_path = Path(args.results)
    else:
        results_path = RESULTS_DIR / "results_all.json"

    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {results_path}")

    # Score all results
    scores = [score_result(r) for r in results]

    # Group by condition
    by_condition = defaultdict(list)
    for s in scores:
        by_condition[s["condition"]].append(s)

    # Print summary table
    print(f"\n{'Condition':<25} {'Parse':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Prec':>6} {'Rec':>6} {'F1':>6} {'CWE':>10} {'Line':>10}")
    print("-" * 115)

    all_metrics = {}
    for cond_name in sorted(by_condition.keys()):
        cond_scores = by_condition[cond_name]
        m = compute_metrics(cond_scores)
        all_metrics[cond_name] = m
        print(f"{cond_name:<25} {m['parse_success']:>8} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['tn']:>5} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['cwe_accuracy']:>10} {m['line_accuracy']:>10}")

    # Save scored results
    scored_path = results_path.parent / results_path.name.replace("results_", "scored_")
    with open(scored_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScored results saved to {scored_path}")

    # Save metrics summary
    metrics_path = results_path.parent / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics summary saved to {metrics_path}")

    # Per-CWE breakdown for each condition
    print("\n\n=== Per-CWE Recall (vulnerable tests only) ===")
    cwes = sorted(set(s["ground_truth_cwe"] for s in scores if s["ground_truth_vulnerable"]))
    header = f"{'CWE':<8}" + "".join(f"{c:<18}" for c in sorted(by_condition.keys()))
    print(header)
    print("-" * len(header))
    for cwe in cwes:
        row = f"CWE-{cwe:<4}"
        for cond_name in sorted(by_condition.keys()):
            cond_vuln = [s for s in by_condition[cond_name]
                         if s["ground_truth_vulnerable"] and s["ground_truth_cwe"] == cwe]
            detected = sum(1 for s in cond_vuln if s["detected_vulnerable"])
            row += f"{detected}/{len(cond_vuln):<16}"
        print(row)


if __name__ == "__main__":
    main()
