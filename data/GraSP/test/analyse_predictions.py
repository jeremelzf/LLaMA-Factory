#!/usr/bin/env python3
"""
analyse_predictions.py
-----------------------
Analyse generated_predictions.jsonl: GT distribution, last-line patterns, and
how often GT labels appear in the model answer (substring / multi-label aware).

Usage:
    python analyse_predictions.py path/to/generated_predictions.jsonl
    python analyse_predictions.py path/to/generated_predictions.jsonl --out analysis_report.txt

Uses the same task prompts and label taxonomy as eval_metrics.py (single source of truth).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

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
    parser.add_argument("predictions_jsonl", type=str,
                        help="Path to generated_predictions.jsonl")
    parser.add_argument("--out", type=str, default=None,
                        help="Write report to this file instead of stdout (avoids encoding issues on Windows)")
    args = parser.parse_args()

    # Output target: file (always UTF-8) or stdout
    if args.out:
        sink = open(args.out, "w", encoding="utf-8")
    else:
        sink = sys.stdout

    def pr(*a, **kw):
        print(*a, **kw, file=sink)

    stats = defaultdict(
        lambda: {
            "total": 0,
            "gt_distribution": Counter(),
            "last_line_patterns": Counter(),
            "correct": 0,
        }
    )

    with open(args.predictions_jsonl, encoding="utf-8") as f:
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

            lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]
            last_line = lines[-1].lower() if lines else ""

            stats[task]["total"] += 1
            if task in MULTILABEL_TASKS:
                parts = parse_gt_multilabel(label_raw)
                stats[task]["gt_distribution"][tuple(sorted(parts))] += 1
                if parts and all(p in answer_lower for p in parts):
                    stats[task]["correct"] += 1
            else:
                label_lower = label_raw.strip().lower()
                stats[task]["gt_distribution"][label_lower] += 1
                if label_lower in answer_lower:
                    stats[task]["correct"] += 1

            stats[task]["last_line_patterns"][last_line[:80]] += 1

    SEP  = "=" * 70
    SEP2 = "-" * 60

    pr(SEP)
    pr("PREDICTION ANALYSIS REPORT")
    pr(SEP)

    for task in ["phase", "step", "instrument", "action"]:
        s = stats[task]
        if s["total"] == 0:
            continue
        pr(f"\n{SEP2}")
        pr(f"TASK: {task.upper()} ({s['total']} samples)")
        pct = s["correct"] / s["total"] * 100
        pr(f"  GT fully in answer (substring / all parts): "
           f"{s['correct']}/{s['total']} = {pct:.1f}%")

        pr(f"\n  Top 10 GROUND TRUTH labels (or label sets):")
        for lbl, cnt in s["gt_distribution"].most_common(10):
            pr(f"    {cnt:5d}x  {lbl}")

        pr(f"\n  Top 20 LAST LINE of model answer:")
        for pat, cnt in s["last_line_patterns"].most_common(20):
            pr(f"    {cnt:5d}x  {pat}")

    pr("\n" + SEP)
    pr("Use this output to tune normalize_single_label / normalize_multilabel.")
    pr(SEP)

    if args.out:
        sink.close()
        print(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()
