#!/usr/bin/env python3
"""
eval_metrics.py
---------------
Evaluates LLaMA-Factory prediction output for the GraSP benchmark.

Expected input (auto-detected):
  - generated_predictions.jsonl: one JSON object per line with keys "predict", "label",
    and optional "prompt" (LLaMA-Factory do_predict). Task type is inferred from question
    text in "prompt" when present; overlapping labels (e.g. phase vs step) use that first.
  - generated_predictions.txt: tab-separated predict\\tlabel\\t... (legacy). Prompt is
    empty, so task falls back to label matching (step before phase for overlaps).

Computes per-task metrics:
  - Phases:          mAP, Macro F1, Classification Accuracy
  - Steps:           mAP, Macro F1, Classification Accuracy
  - Instruments:     Presence-based mAP, Macro F1, Classification Accuracy
  - Atomic Actions:  Presence-based mAP, Macro F1, Classification Accuracy
"""

import argparse
import difflib
import json
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    average_precision_score,
    classification_report
)
from sklearn.preprocessing import LabelBinarizer


# ─────────────────────────────────────────────
# 1. Load predictions file
# ─────────────────────────────────────────────

def _sniff_is_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            return s.startswith("{")
    return False


def load_predictions_tsv(filepath):
    """Tab-separated: predict\\tlabel\\t... (raw strings, not lowercased)."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            samples.append((parts[0].strip(), parts[1].strip(), ""))
    return samples


def load_predictions_jsonl(filepath):
    """
    LLaMA-Factory JSONL: each line is JSON with "predict" and "label"; optional "prompt"
    is used for task detection when present.
    """
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{filepath}:{lineno}: invalid JSON") from e
            if "predict" not in obj or "label" not in obj:
                raise KeyError(
                    f"{filepath}:{lineno}: expected keys 'predict' and 'label', got {list(obj.keys())}"
                )
            samples.append(
                (str(obj["predict"]), str(obj["label"]), str(obj.get("prompt", "")))
            )
    return samples


def load_predictions(filepath):
    """
    Load predictions from JSONL or TSV.
    Returns list of (predict, label, prompt) as raw strings (prompt may be "").
    """
    path_lower = filepath.lower()
    if path_lower.endswith(".jsonl") or _sniff_is_jsonl(filepath):
        return load_predictions_jsonl(filepath)
    return load_predictions_tsv(filepath)


# Text after common chain-of-thought / thinking markers (model-specific).
_THINK_END_MARKERS = (
    "</think>",
)


def extract_answer_text(pred: str) -> str:
    """Keep the final answer span; drop preceding reasoning if delimited."""
    text = pred.strip()
    for marker in _THINK_END_MARKERS:
        if marker in text:
            text = text.split(marker)[-1]
    return text.strip()


def _token_recall(candidate: str, haystack_tokens: set) -> float:
    ct = set(re.findall(r"[a-z0-9]+", candidate.lower()))
    if not ct:
        return 0.0
    return len(ct & haystack_tokens) / len(ct)


def _phrase_in_haystack(phrase: str, haystack: str) -> bool:
    """Substring match with word boundaries for short labels (avoid 'open' in 'opening')."""
    if phrase not in haystack:
        return False
    if len(phrase) <= 6:
        return (
            re.search(
                r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])",
                haystack,
            )
            is not None
        )
    return True


def normalize_prediction(pred: str, task: str) -> str:
    """
    Map free-form model output to a single canonical label for this task.

    Uses, in order: longest substring match against task labels, then token-recall
    against the answer span, then difflib against the last substantial line.
    Always returns a member of the task label set when task is known.
    """
    if task not in TASK_TO_LABELS:
        return " ".join(pred.strip().lower().split())

    candidates = TASK_TO_LABELS[task]
    answer = extract_answer_text(pred)
    haystack = " ".join(answer.lower().split())
    hay_tokens = set(re.findall(r"[a-z0-9]+", haystack))

    best_sub = None
    best_len = -1
    for c in candidates:
        if _phrase_in_haystack(c, haystack) and len(c) > best_len:
            best_sub = c
            best_len = len(c)
    if best_sub is not None:
        return best_sub

    scored = [(_token_recall(c, hay_tokens), c) for c in candidates]
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    best_score, best_c = scored[0]
    if best_score >= 0.72:
        return best_c

    tail_lines = [ln.strip() for ln in answer.split("\n") if len(ln.strip()) > 12]
    if tail_lines:
        last = tail_lines[-1].lower()
        last = " ".join(last.split())
        close = difflib.get_close_matches(last, list(candidates), n=1, cutoff=0.55)
        if close:
            return close[0]

    if best_score >= 0.45:
        return best_c

    return scored[0][1]


# ─────────────────────────────────────────────
# 2. Task detection (prompt question text, then label fallback)
# ─────────────────────────────────────────────

# Substrings from GraSP / LLaMA-Factory templates; must match training/eval prompts.
_TASK_PROMPT_SNIPPETS = (
    ("phase", "what surgical phase"),
    ("step", "what surgical step"),
    ("instrument", "what surgical instruments"),
    ("action", "what atomic actions"),
)

# GraSP phase labels (lowercase)
PHASE_LABELS = {
    "idle",
    "left pelvic isolated lymphadenectomy",
    "right pelvic isolated lymphadenectomy",
    "developing the space of retzius",
    "ligation of the deep dorsal venous complex",
    "bladder neck identification and transection",
    "seminal vesicle dissection",
    "development of the plane between the prostate and rectum",
    "prostatic pedicle control",
    "severing of the prostate from the urethra",
    "bladder neck reconstruction",
}

# GraSP step labels (lowercase)
STEP_LABELS = {
    "idle",
    "identification and dissection of the iliac vein and artery",
    "cutting and dissection of the external iliac veins lymph node",
    "obturator nerve and vessel path identification, dissection and cutting of the obturator lymph nodes",
    "insert the lymph nodes in retrieval bags",
    "prevessical dissection",
    "ligation of the dorsal venous complex",
    "prostate dissection until the levator ani",
    "seminal vesicle dissection",
    "dissection of denonviliers fascia",
    "cut the tissue between the prostate and the urethra",
    "hold prostate",
    "insert prostate in retrieval bag",
    "pass suture to the urethra",
    "pass suture to the bladder neck",
    "pull suture",
    "tie suture",
    "suction",
    "cut suture or tissue",
    "cut between the prostate and bladder neck",
    "vascular pedicle control",
}

# GraSP instrument labels (lowercase)
INSTRUMENT_LABELS = {
    "bipolar forceps", "prograsp forceps", "large needle driver",
    "monopolar curved scissors", "suction instrument",
    "clip applier", "laparoscopic grasper",
}

# GraSP atomic action labels (lowercase)
ACTION_LABELS = {
    "cauterize", "close", "cut", "grasp", "hold",
    "open", "open something", "pull", "push", "release",
    "still", "suction", "travel", "other"
}

TASK_TO_LABELS = {
    "phase": PHASE_LABELS,
    "step": STEP_LABELS,
    "instrument": INSTRUMENT_LABELS,
    "action": ACTION_LABELS,
}


def detect_task(label: str, prompt: str = "") -> str:
    """
    Detect GraSP task: use question text in prompt when present, else label sets.
    Returns one of: 'phase', 'step', 'instrument', 'action', 'unknown'.

    Label fallback uses step before phase so overlapping names (e.g. idle, seminal
    vesicle dissection) map to step when prompt is missing.
    """
    pl = prompt.lower()
    for task_name, snippet in _TASK_PROMPT_SNIPPETS:
        if snippet in pl:
            return task_name

    l = label.strip().lower()
    if l in STEP_LABELS:
        return "step"
    if l in PHASE_LABELS:
        return "phase"
    if l in INSTRUMENT_LABELS:
        return "instrument"
    if l in ACTION_LABELS:
        return "action"
    return "unknown"


# ─────────────────────────────────────────────
# 3. Compute metrics for one task
# ─────────────────────────────────────────────

def compute_metrics(preds, labels, task_name):
    """
    Computes Accuracy, Macro F1, and presence-based mAP for one task.
    """
    print(f"\n{'='*60}")
    print(f"  Task: {task_name.upper()}  ({len(labels)} samples)")
    print(f"{'='*60}")

    # Accuracy
    accuracy = accuracy_score(labels, preds)
    print(f"  Classification Accuracy : {accuracy*100:.2f}%")

    # Macro F1
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"  Macro F1-score          : {macro_f1*100:.2f}%")

    # Presence-based mAP
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(labels)
    classes = lb.classes_

    y_score = np.zeros_like(y_true_bin, dtype=float)
    pred_idx = {c: i for i, c in enumerate(classes)}
    for i, p in enumerate(preds):
        if p in pred_idx:
            y_score[i, pred_idx[p]] = 1.0

    if y_true_bin.shape[1] == 1:
        print(f"  mAP                     : N/A (only 1 class present)")
        mAP = float("nan")
    else:
        per_class_ap = []
        for c in range(y_true_bin.shape[1]):
            if y_true_bin[:, c].sum() == 0:
                continue
            ap = average_precision_score(y_true_bin[:, c], y_score[:, c])
            per_class_ap.append(ap)
        mAP = float(np.mean(per_class_ap)) if per_class_ap else float("nan")
        print(f"  mAP (presence-based)    : {mAP*100:.2f}%")

    # Per-class breakdown
    print(f"\n  Per-class F1 breakdown:")
    report = classification_report(labels, preds, zero_division=0)
    for line in report.split("\n"):
        print("    " + line)

    return {
        "task":      task_name,
        "n_samples": len(labels),
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
        "mAP":       mAP,
    }


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute GraSP evaluation metrics from LLaMA-Factory predictions."
    )
    parser.add_argument(
        "predictions_file",
        type=str,
        help="Path to generated_predictions.jsonl or .txt (LLaMA-Factory do_predict)."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="eval_results.json",
        help="Path to save JSON summary of results (default: eval_results.json)."
    )
    args = parser.parse_args()

    print(f"\nLoading predictions from: {args.predictions_file}")
    samples = load_predictions(args.predictions_file)
    print(f"Total samples loaded: {len(samples)}")

    # Split samples by task
    task_data = defaultdict(lambda: {"preds": [], "labels": []})
    unknown_count = 0

    for pred, label, prompt in samples:
        label_clean = label.strip().lower()
        task = detect_task(label_clean, prompt=prompt)
        if task == "unknown":
            unknown_count += 1
            continue
        pred_norm = normalize_prediction(pred, task)
        task_data[task]["preds"].append(pred_norm)
        task_data[task]["labels"].append(label_clean)

    if unknown_count:
        print(f"\nWarning: {unknown_count} samples could not be assigned to a task and were skipped.")
        print("  Tip: Check if your labels exactly match the GraSP label names in the script.")

    # Compute metrics per task
    all_results = []
    for task_name in ["phase", "step", "instrument", "action"]:
        data = task_data.get(task_name)
        if not data or len(data["labels"]) == 0:
            print(f"\nNo samples found for task: {task_name} — skipping.")
            continue
        result = compute_metrics(data["preds"], data["labels"], task_name)
        all_results.append(result)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<15} {'N':>8} {'Accuracy':>10} {'Macro F1':>10} {'mAP':>10}")
    print(f"  {'-'*53}")
    for r in all_results:
        mAP_str = f"{r['mAP']*100:.2f}%" if not np.isnan(r["mAP"]) else "N/A"
        print(
            f"  {r['task']:<15} {r['n_samples']:>8} "
            f"{r['accuracy']*100:>9.2f}% "
            f"{r['macro_f1']*100:>9.2f}% "
            f"{mAP_str:>10}"
        )

    # Save JSON
    def safe(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v

    json_out = [
        {**r, "accuracy": safe(r["accuracy"]),
               "macro_f1": safe(r["macro_f1"]),
               "mAP":      safe(r["mAP"])}
        for r in all_results
    ]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to: {args.output_json}\n")


if __name__ == "__main__":
    main()
