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

Metrics:
  - Phase & step (single-label): classification accuracy, macro F1, presence-based mAP.
  - Instrument & action (multi-label GT): exact set match rate, macro / micro F1, mAP
    (MultiLabelBinarizer over the fixed GraSP label set for that task).
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
    classification_report,
)
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


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
    "</redacted_thinking>",
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


def _get_last_sentence(answer: str) -> str:
    """Final sentence by punctuation, else last non-empty line."""
    t = answer.strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts[-1]
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return lines[-1] if lines else t


def normalize_single_label(pred: str, task: str) -> str:
    """
    Single-label (phase / step): longest phrase in full answer, then last-sentence
    fuzzy match, then token-recall (no hardcoded class fallback).
    """
    if task not in TASK_TO_LABELS:
        return " ".join(pred.strip().lower().split())

    candidates = TASK_TO_LABELS[task]
    answer = extract_answer_text(pred)
    haystack = " ".join(answer.lower().split())
    hay_tokens = set(re.findall(r"[a-z0-9]+", haystack))

    best_sub, best_len = None, -1
    for c in candidates:
        if _phrase_in_haystack(c, haystack) and len(c) > best_len:
            best_sub, best_len = c, len(c)
    if best_sub is not None:
        return best_sub

    last_sent = _get_last_sentence(answer)
    hay_last = " ".join(last_sent.lower().split())
    if hay_last:
        best_sub, best_len = None, -1
        for c in candidates:
            if _phrase_in_haystack(c, hay_last) and len(c) > best_len:
                best_sub, best_len = c, len(c)
        if best_sub is not None:
            return best_sub
        close = difflib.get_close_matches(
            hay_last, list(candidates), n=1, cutoff=0.55
        )
        if close:
            return close[0]

    scored = [(_token_recall(c, hay_tokens), c) for c in candidates]
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    best_score, best_c = scored[0]

    if best_score >= 0.72:
        return best_c

    tail_lines = [ln.strip() for ln in answer.split("\n") if len(ln.strip()) > 12]
    if tail_lines:
        last = " ".join(tail_lines[-1].lower().split())
        close = difflib.get_close_matches(last, list(candidates), n=1, cutoff=0.55)
        if close:
            return close[0]

    if best_score >= 0.45:
        return best_c

    return scored[0][1]


def normalize_multilabel(pred: str, task: str) -> frozenset:
    """
    Instrument / action: collect every GraSP label supported by the answer — phrase
    match first, then token-recall >= 0.60. If none matched, use best single
    token-recall label (same family as single-label fallback).
    """
    candidates = TASK_TO_LABELS[task]
    answer = extract_answer_text(pred)
    haystack = " ".join(answer.lower().split())
    hay_tokens = set(re.findall(r"[a-z0-9]+", haystack))

    matched = {c for c in candidates if _phrase_in_haystack(c, haystack)}
    for c in candidates:
        if c not in matched and _token_recall(c, hay_tokens) >= 0.60:
            matched.add(c)

    if matched:
        return frozenset(matched)

    scored = [(_token_recall(c, hay_tokens), c) for c in candidates]
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    return frozenset({scored[0][1]})


def parse_gt_multilabel(label_raw: str) -> frozenset:
    """Split comma/semicolon-separated GT into normalized lowercase labels."""
    parts = re.split(r"[,;]", label_raw)
    return frozenset(p.strip().lower() for p in parts if p.strip())


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

MULTILABEL_TASKS = frozenset({"instrument", "action"})


def detect_task(label: str, prompt: str = "") -> str:
    """
    Detect GraSP task: use question text in prompt when present, else label sets.
    Returns one of: 'phase', 'step', 'instrument', 'action', 'unknown'.

    Label fallback: comma-separated lists of instruments or actions are detected
    before single-label overlap resolution (step before phase).
    """
    pl = prompt.lower()
    for task_name, snippet in _TASK_PROMPT_SNIPPETS:
        if snippet in pl:
            return task_name

    l = label.strip().lower()
    if "," in l or ";" in l:
        parts = [p.strip().lower() for p in re.split(r"[,;]", l) if p.strip()]
        if parts and all(p in INSTRUMENT_LABELS for p in parts):
            return "instrument"
        if parts and all(p in ACTION_LABELS for p in parts):
            return "action"

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
# 3. Metrics
# ─────────────────────────────────────────────

def compute_metrics_single(preds, labels, task_name):
    """Phase / step: accuracy, macro F1, presence-based mAP."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_name.upper()}  ({len(labels)} samples)  [single-label]")
    print(f"{'='*60}")

    accuracy = accuracy_score(labels, preds)
    print(f"  Classification Accuracy : {accuracy*100:.2f}%")

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"  Macro F1-score          : {macro_f1*100:.2f}%")

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

    print(f"\n  Per-class F1 breakdown:")
    report = classification_report(labels, preds, zero_division=0)
    for line in report.split("\n"):
        print("    " + line)

    return {
        "task": task_name,
        "n_samples": len(labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mAP": mAP,
        "exact_match": None,
        "micro_f1": None,
    }


def compute_metrics_multilabel(preds_sets, labels_sets, task_name):
    """Instrument / action: exact match, macro/micro F1, mAP."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_name.upper()}  ({len(labels_sets)} samples)  [multi-label]")
    print(f"{'='*60}")

    classes = sorted(TASK_TO_LABELS[task_name])
    mlb = MultiLabelBinarizer(classes=classes)
    y_true = mlb.fit_transform([list(s) for s in labels_sets])
    y_pred = mlb.transform([list(s) for s in preds_sets])

    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    print(f"  Exact match (set)       : {exact*100:.2f}%")

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"  Macro F1-score          : {macro_f1*100:.2f}%")
    print(f"  Micro F1-score          : {micro_f1*100:.2f}%")

    if y_true.shape[1] <= 1:
        print(f"  mAP                     : N/A (<=1 class column)")
        mAP = float("nan")
    else:
        try:
            mAP = float(average_precision_score(y_true, y_pred, average="macro"))
        except ValueError:
            mAP = float("nan")
        if not np.isnan(mAP):
            print(f"  mAP (macro)             : {mAP*100:.2f}%")
        else:
            print(f"  mAP (macro)             : N/A")

    print(f"\n  Per-label F1 (micro-averaged report):")
    report = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0
    )
    for line in report.split("\n"):
        print("    " + line)

    return {
        "task": task_name,
        "n_samples": len(labels_sets),
        "accuracy": None,
        "exact_match": exact,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mAP": mAP,
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
        nargs="?",
        help="Path to generated_predictions.jsonl or .txt (LLaMA-Factory do_predict).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="eval_results.json",
        help="Path to save JSON summary of results (default: eval_results.json).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in spot checks from cursor_prompt.md and exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    print(f"\nLoading predictions from: {args.predictions_file}")
    samples = load_predictions(args.predictions_file)
    print(f"Total samples loaded: {len(samples)}")

    task_data = defaultdict(
        lambda: {"preds": [], "labels": [], "multilabel": False}
    )
    unknown_count = 0

    for pred, label, prompt in samples:
        task = detect_task(label, prompt=prompt)
        if task == "unknown":
            unknown_count += 1
            continue

        if task in MULTILABEL_TASKS:
            task_data[task]["multilabel"] = True
            gt_set = parse_gt_multilabel(label)
            pred_set = normalize_multilabel(pred, task)
            task_data[task]["preds"].append(pred_set)
            task_data[task]["labels"].append(gt_set)
        else:
            label_clean = label.strip().lower()
            pred_norm = normalize_single_label(pred, task)
            task_data[task]["preds"].append(pred_norm)
            task_data[task]["labels"].append(label_clean)

    if unknown_count:
        print(f"\nWarning: {unknown_count} samples could not be assigned to a task and were skipped.")
        print("  Tip: Check if your labels exactly match the GraSP label names in the script.")

    all_results = []
    for task_name in ["phase", "step", "instrument", "action"]:
        data = task_data.get(task_name)
        if not data or len(data["labels"]) == 0:
            print(f"\nNo samples found for task: {task_name} — skipping.")
            continue
        if data["multilabel"]:
            result = compute_metrics_multilabel(
                data["preds"], data["labels"], task_name
            )
        else:
            result = compute_metrics_single(
                data["preds"], data["labels"], task_name
            )
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    hdr = f"  {'Task':<15} {'N':>8} {'Acc/Ex':>10} {'MacroF1':>10} {'MicroF1':>10} {'mAP':>10}"
    print(hdr)
    print(f"  {'-'*63}")
    for r in all_results:
        mAP_str = f"{r['mAP']*100:.2f}%" if r["mAP"] is not None and not np.isnan(r["mAP"]) else "N/A"
        if r["exact_match"] is not None:
            acc_ex = f"{r['exact_match']*100:.2f}%"
            micro_str = f"{r['micro_f1']*100:.2f}%" if r["micro_f1"] is not None else "N/A"
        else:
            acc_ex = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
            micro_str = "—"
        print(
            f"  {r['task']:<15} {r['n_samples']:>8} "
            f"{acc_ex:>10} "
            f"{r['macro_f1']*100:>9.2f}% "
            f"{micro_str:>10} "
            f"{mAP_str:>10}"
        )

    def safe(v):
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    json_out = []
    for r in all_results:
        row = {
            "task": r["task"],
            "n_samples": r["n_samples"],
            "macro_f1": safe(r["macro_f1"]),
            "mAP": safe(r["mAP"]),
        }
        if r["accuracy"] is not None:
            row["accuracy"] = safe(r["accuracy"])
        if r["exact_match"] is not None:
            row["exact_match"] = safe(r["exact_match"])
        if r["micro_f1"] is not None:
            row["micro_f1"] = safe(r["micro_f1"])
        json_out.append(row)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to: {args.output_json}\n")


def _run_self_test():
    """Spot checks aligned with cursor_prompt.md."""
    tests = [
        {
            "task": "phase",
            "predict": "<redacted_thinking>...reasoning...</redacted_thinking>\nThe surgical phase is currently idle, with instruments positioned but not actively manipulating tissue.",
            "label": "Left pelvic isolated lymphadenectomy",
        },
        {
            "task": "step",
            "predict": "<redacted_thinking>...reasoning...</redacted_thinking>\nThe surgical step is currently idle, with instruments positioned for retraction.",
            "label": "Insert the lymph nodes in retrieval bags",
        },
        {
            "task": "instrument",
            "predict": "<redacted_thinking>...reasoning...</redacted_thinking>\nThe surgical instruments visible are a grasper and a cautery probe.",
            "label": "Bipolar Forceps, Monopolar Curved Scissors",
        },
        {
            "task": "action",
            "predict": "<redacted_thinking>...reasoning...</redacted_thinking>\nThe atomic actions are tensioning adjacent tissue, stabilizing the field, and ligating a vascular structure with suture.",
            "label": "Hold, Push, Still, Travel",
        },
    ]
    print("Self-test (extraction only; task forced):\n")
    for t in tests:
        task = t["task"]
        if task in MULTILABEL_TASKS:
            gt = parse_gt_multilabel(t["label"])
            pr = normalize_multilabel(t["predict"], task)
            print(f"  [{task}] GT set: {sorted(gt)}")
            print(f"  [{task}] Pred set: {sorted(pr)}")
        else:
            gt = t["label"].strip().lower()
            pr = normalize_single_label(t["predict"], task)
            print(f"  [{task}] GT: {gt}")
            print(f"  [{task}] Pred: {pr}")
        print()


if __name__ == "__main__":
    main()
