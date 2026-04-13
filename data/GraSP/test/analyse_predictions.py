#!/usr/bin/env python3
"""
Analyse generated_predictions.jsonl: GT distribution, last-line patterns, and
how often GT labels appear in the model answer (substring / multi-label aware).

Usage:
  python analyse_predictions.py path/to/generated_predictions.jsonl
  python analyse_predictions.py path/to/generated_predictions.jsonl > analysis_report.txt

Uses the same task prompts and label taxonomy as eval_metrics.py (single source of truth).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Same directory as eval_metrics.py
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_metrics import (  # noqa: E402
    MULTILABEL_TASKS,
    _TASK_PROMPT_SNIPPETS,
    extract_answer_text,
    parse_gt_multilabel,
)


def detect_task_from_prompt(prompt: str) -> str:
    pl = prompt.lower()
    for name, snip in _TASK_PROMPT_SNIPPETS:
        if snip in pl:
            return name
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Analyse GraSP prediction JSONL.")
    parser.add_argument(
        "predictions_jsonl",
        type=str,
        help="Path to generated_predictions.jsonl",
    )
    args = parser.parse_args()
    path = args.predictions_jsonl

    stats = defaultdict(
        lambda: {
            "total": 0,
            "gt_distribution": Counter(),
            "last_line_patterns": Counter(),
            "correct": 0,
        }
    )

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = detect_task_from_prompt(obj.get("prompt", ""))
            if task == "unknown":
                continue

            pred_text = obj["predict"]
            label_raw = obj["label"]
            answer = extract_answer_text(pred_text)
            answer_lower = answer.lower()

            lines = [l.strip() for l in answer.split("\n") if l.strip()]
            last_line = lines[-1].lower() if lines else ""

            stats[task]["total"] += 1
            if task in MULTILABEL_TASKS:
                parts = parse_gt_multilabel(label_raw)
                stats[task]["gt_distribution"][tuple(sorted(parts))] += 1
                found_all = parts and all(p in answer_lower for p in parts)
                if found_all:
                    stats[task]["correct"] += 1
            else:
                label_lower = label_raw.strip().lower()
                stats[task]["gt_distribution"][label_lower] += 1
                if label_lower in answer_lower:
                    stats[task]["correct"] += 1

            stats[task]["last_line_patterns"][last_line[:80]] += 1

    print("=" * 70)
    print("PREDICTION ANALYSIS REPORT")
    print("=" * 70)

    for task in ["phase", "step", "instrument", "action"]:
        s = stats[task]
        if s["total"] == 0:
            continue
        print(f"\n{'─' * 60}")
        print(f"TASK: {task.upper()}  ({s['total']} samples)")
        print(
            f"  GT fully in answer (substring / all parts): "
            f"{s['correct']}/{s['total']} = {s['correct']/s['total']*100:.1f}%"
        )

        print(f"\n  Top 10 GROUND TRUTH labels (or label sets):")
        for lbl, cnt in s["gt_distribution"].most_common(10):
            print(f"    {cnt:5d}x  {lbl}")

        print(f"\n  Top 20 LAST LINE of model answer:")
        for pat, cnt in s["last_line_patterns"].most_common(20):
            print(f"    {cnt:5d}x  {pat}")

    print("\n" + "=" * 70)
    print("Use this output to tune normalize_single_label / normalize_multilabel.")
    print("=" * 70)


if __name__ == "__main__":
    main()
