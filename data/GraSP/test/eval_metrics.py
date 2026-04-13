#!/usr/bin/env python3
"""
eval_metrics.py  (v4 — GraSP co-occurrence constraint rules)
-------------------------------------------------------------
Pipeline per multi-label prediction (instrument / action):

  STEP 1 — Synonym expansion  (additive)
            phrase match → token recall → _expand_*_synonyms()

  STEP 2 — GraSP constraint pruning  (subtractive)
            Runs AFTER synonym expansion. Never overwrites a synonym
            result; only removes physically impossible labels.

            Rules implemented (Ayobi et al., Figs C.13–C.16, Sec 3.1):
            (a) Instrument × Action impossibility
            (b) Step-specific instrument must_include / must_exclude
            (c) Action co-occurrence impossibility
            (d) Phase → valid steps hierarchy check (single-label step)
"""

import argparse, difflib, json, re, sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    f1_score, accuracy_score, average_precision_score, classification_report,
)
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GraSP Label Sets
# ─────────────────────────────────────────────────────────────────────────────

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

INSTRUMENT_LABELS = {
    "bipolar forceps",
    "prograsp forceps",
    "large needle driver",
    "monopolar curved scissors",
    "suction instrument",
    "clip applier",
    "laparoscopic grasper",
}

ACTION_LABELS = {
    "cauterize", "close", "cut", "grasp", "hold",
    "open", "open something", "pull", "push", "release",
    "still", "suction", "travel", "other",
}

TASK_TO_LABELS = {
    "phase":      PHASE_LABELS,
    "step":       STEP_LABELS,
    "instrument": INSTRUMENT_LABELS,
    "action":     ACTION_LABELS,
}

MULTILABEL_TASKS = frozenset({"instrument", "action"})

_TASK_PROMPT_SNIPPETS = (
    ("phase",      "what surgical phase"),
    ("step",       "what surgical step"),
    ("instrument", "what surgical instruments"),
    ("action",     "what atomic actions"),
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Synonym / Paraphrase Maps
# ─────────────────────────────────────────────────────────────────────────────

_STEP_SYNONYMS = {
    "prevesical dissection":                                "prevessical dissection",
    "prevesical":                                           "prevessical dissection",
    "pulling the suture":                                   "pull suture",
    "pulling a suture":                                     "pull suture",
    "pull the suture":                                      "pull suture",
    "passing a suture through the bladder neck":            "pass suture to the bladder neck",
    "passing a suture through the urethra":                 "pass suture to the urethra",
    "passing suture through the bladder neck":              "pass suture to the bladder neck",
    "passing suture through the urethra":                   "pass suture to the urethra",
    "pass a suture through the bladder neck":               "pass suture to the bladder neck",
    "pass a suture through the urethra":                    "pass suture to the urethra",
    "tying the suture":                                     "tie suture",
    "tying a suture":                                       "tie suture",
    "tie the suture":                                       "tie suture",
    "cutting a suture":                                     "cut suture or tissue",
    "cutting suture":                                       "cut suture or tissue",
    "cutting tissue":                                       "cut suture or tissue",
    "dissection and mobilization of the seminal vesicle":   "seminal vesicle dissection",
    "dissecting the seminal vesicle":                       "seminal vesicle dissection",
    "mobilization of the seminal vesicle":                  "seminal vesicle dissection",
    "dissection of the seminal vesicle":                    "seminal vesicle dissection",
    "cutting and dissection of lymph node":                 "cutting and dissection of the external iliac veins lymph node",
    "dissection of the iliac vein":                         "identification and dissection of the iliac vein and artery",
    "dissecting the iliac vein":                            "identification and dissection of the iliac vein and artery",
    "denonvilliers fascia":                                 "dissection of denonviliers fascia",
    "denonvilier fascia":                                   "dissection of denonviliers fascia",
    "denonviliers":                                         "dissection of denonviliers fascia",
    "vascular pedicle":                                     "vascular pedicle control",
    "prostatic pedicle":                                    "vascular pedicle control",
    "dorsal venous complex":                                "ligation of the dorsal venous complex",
    "ligation of the dorsal vein":                          "ligation of the dorsal venous complex",
}

_INSTRUMENT_SYNONYMS = {
    "laparoscopic grasper":         "laparoscopic grasper",
    "grasper":                      "laparoscopic grasper",
    "prograsp":                     "prograsp forceps",
    "pro-grasp":                    "prograsp forceps",
    "bipolar":                      "bipolar forceps",
    "bipolar forceps":              "bipolar forceps",
    "maryland forceps":             "bipolar forceps",
    "fenestrated forceps":          "bipolar forceps",
    "monopolar":                    "monopolar curved scissors",
    "monopolar curved scissors":    "monopolar curved scissors",
    "monopolar scissors":           "monopolar curved scissors",
    "curved scissors":              "monopolar curved scissors",
    "monopolar hook":               "monopolar curved scissors",
    "dissecting energy device":     "monopolar curved scissors",
    "dissector":                    "monopolar curved scissors",
    "cautery":                      "bipolar forceps",
    "cautery probe":                "bipolar forceps",
    "electrocautery":               "bipolar forceps",
    "needle driver":                "large needle driver",
    "needle holder":                "large needle driver",
    "large needle driver":          "large needle driver",
    "suction instrument":           "suction instrument",
    "suction":                      "suction instrument",
    "irrigator":                    "suction instrument",
    "irrigation":                   "suction instrument",
    "suction/irrigation":           "suction instrument",
    "clip applier":                 "clip applier",
    "clip":                         "clip applier",
    "clips":                        "clip applier",
}

_ACTION_SYNONYMS = {
    "holding":              "hold",
    "stabilizing":          "hold",
    "stabilization":        "hold",
    "grasping":             "hold",
    "retracting":           "pull",
    "retraction":           "pull",
    "pulling":              "pull",
    "traction":             "pull",
    "pushing":              "push",
    "advancing":            "push",
    "passing":              "push",
    "inserting":            "push",
    "cutting":              "cut",
    "transecting":          "cut",
    "incising":             "cut",
    "dissecting":           "cut",
    "dissection":           "cut",
    "trimming":             "cut",
    "cauterizing":          "cauterize",
    "coagulating":          "cauterize",
    "coagulation":          "cauterize",
    "electrosurgical":      "cauterize",
    "cautery":              "cauterize",
    "idle":                 "still",
    "stationary":           "still",
    "no active":            "still",
    "positioned but not":   "still",
    "not actively":         "still",
    "suctioning":           "suction",
    "aspirating":           "suction",
    "aspiration":           "suction",
    "irrigation":           "suction",
    "traveling":            "travel",
    "repositioning":        "travel",
    "moving":               "travel",
    "withdrawing":          "travel",
    "inserting instrument": "travel",
    "gripping":             "grasp",
    "closing":              "close",
    "approximating":        "close",
    "opening":              "open",
    "spreading":            "open",
    "exposing":             "open",
    "releasing":            "release",
    "opening something":    "open something",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  GraSP Co-occurrence Constraint Tables
#     Source: Ayobi et al., Figs C.13-C.16, Section 3.1
# ─────────────────────────────────────────────────────────────────────────────

# 3a. Instrument -> physically allowed actions (Fig C.13)
_INSTRUMENT_ALLOWED_ACTIONS = {
    "monopolar curved scissors": {
        "travel", "still", "hold", "push", "pull",
        "open", "close", "cut", "cauterize",
    },
    "bipolar forceps": {
        "travel", "still", "hold", "push", "pull",
        "open", "close", "grasp", "release",
        "cauterize", "open something",
    },
    "prograsp forceps": {
        "travel", "still", "hold", "push", "pull",
        "open", "close", "grasp", "release", "open something",
    },
    "large needle driver": {
        "travel", "still", "hold", "push", "pull",
        "close", "grasp", "release",
    },
    "suction instrument": {
        "travel", "still", "suction", "push",
    },
    "laparoscopic grasper": {
        "travel", "still", "hold", "push", "pull",
        "open", "close", "grasp", "release",
        "suction", "open something",
    },
    "clip applier": {
        "travel", "still", "close", "push", "other",
    },
}

# 3b. Action pairs that CANNOT co-occur on the same instrument (Fig C.13 top)
_INCOMPATIBLE_ACTION_PAIRS = frozenset({
    frozenset({"cut",       "suction"}),
    frozenset({"cut",       "grasp"}),
    frozenset({"cut",       "release"}),
    frozenset({"cut",       "open something"}),
    frozenset({"cauterize", "suction"}),
    frozenset({"cauterize", "grasp"}),
    frozenset({"suction",   "hold"}),
    frozenset({"suction",   "pull"}),
    frozenset({"suction",   "cut"}),
    frozenset({"suction",   "cauterize"}),
    frozenset({"still",     "cut"}),
    frozenset({"still",     "cauterize"}),
    frozenset({"still",     "pull"}),
    frozenset({"still",     "push"}),
    frozenset({"still",     "open"}),
    frozenset({"still",     "close"}),
})

# 3c. Step-specific instrument constraints (Section 3.1 + Fig C.15)
_STEP_INSTRUMENT_CONSTRAINTS = {
    "pass suture to the urethra":       {"must_exclude": {"monopolar curved scissors"}},
    "pass suture to the bladder neck":  {"must_exclude": {"monopolar curved scissors"}},
    "pull suture":                      {"must_exclude": {"monopolar curved scissors"}},
    "tie suture":                       {"must_exclude": {"monopolar curved scissors"}},
    "vascular pedicle control":         {"must_include": {"clip applier"}},
    "suction":                          {"must_include": {"suction instrument"},
                                         "must_exclude": {"clip applier"}},
    "hold prostate":                    {"must_exclude": {"clip applier",
                                                           "suction instrument"}},
    "insert prostate in retrieval bag": {"must_exclude": {"monopolar curved scissors",
                                                           "bipolar forceps",
                                                           "clip applier"}},
    "insert the lymph nodes in retrieval bags": {
        "must_exclude": {"monopolar curved scissors", "bipolar forceps", "clip applier"}},
}

# 3d. Phase -> valid steps (Fig B.2 hierarchy, medium confidence)
_PHASE_TO_VALID_STEPS = {
    "left pelvic isolated lymphadenectomy": {
        "identification and dissection of the iliac vein and artery",
        "cutting and dissection of the external iliac veins lymph node",
        "obturator nerve and vessel path identification, dissection and cutting of the obturator lymph nodes",
        "insert the lymph nodes in retrieval bags", "idle",
    },
    "right pelvic isolated lymphadenectomy": {
        "identification and dissection of the iliac vein and artery",
        "cutting and dissection of the external iliac veins lymph node",
        "obturator nerve and vessel path identification, dissection and cutting of the obturator lymph nodes",
        "insert the lymph nodes in retrieval bags", "idle",
    },
    "developing the space of retzius":              {"prevessical dissection", "idle"},
    "ligation of the deep dorsal venous complex":   {"ligation of the dorsal venous complex", "idle"},
    "bladder neck identification and transection":  {"cut between the prostate and bladder neck", "idle"},
    "seminal vesicle dissection":                   {"seminal vesicle dissection", "idle"},
    "development of the plane between the prostate and rectum": {
        "dissection of denonviliers fascia", "prostate dissection until the levator ani", "idle"},
    "prostatic pedicle control":                    {"vascular pedicle control", "idle"},
    "severing of the prostate from the urethra":    {
        "cut the tissue between the prostate and the urethra", "idle"},
    "bladder neck reconstruction": {
        "hold prostate", "insert prostate in retrieval bag",
        "pass suture to the urethra", "pass suture to the bladder neck",
        "pull suture", "tie suture", "suction", "cut suture or tissue", "idle",
    },
    "idle": {"idle"},
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Energy-Device Resolution  (context-aware)
# ─────────────────────────────────────────────────────────────────────────────

_RECONSTRUCTIVE_STEPS = {
    "pass suture", "pull suture", "tie suture",
    "bladder neck reconstruction", "insert prostate in retrieval bag",
}

def resolve_energy_device(haystack, action_tokens, predicted_context=""):
    ctx = predicted_context.lower()
    if "idle" in ctx:
        return None
    cut_kw     = {"cut","cutting","dissecting","dissection","incising","transecting"}
    cautery_kw = {"cauterize","cauterizing","coagulating","coagulation"}
    has_cut     = bool(action_tokens & cut_kw)
    has_cautery = bool(action_tokens & cautery_kw)
    if has_cut and not has_cautery:     return "monopolar curved scissors"
    if has_cautery and not has_cut:     return "bipolar forceps"
    if has_cut and has_cautery:         return "monopolar curved scissors"
    if any(s in ctx for s in _RECONSTRUCTIVE_STEPS): return "bipolar forceps"
    dissect_kws = {"lymphadenectomy","dissection","pedicle","seminal vesicle",
                   "bladder neck","dorsal venous","levator ani","denonviliers",
                   "retzius","obturator","iliac"}
    if any(d in ctx for d in dissect_kws): return "monopolar curved scissors"
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Constraint Functions  (STEP 2 - subtractive)
# ─────────────────────────────────────────────────────────────────────────────

def apply_instrument_action_constraints(instruments, actions, step_context=""):
    """
    (a) Prune actions impossible for every predicted instrument.
    (b) Prune instruments whose allowed set is disjoint from predicted actions.
    (c) Step-specific must_include / must_exclude rules.
    Never empties either set.
    """
    if not instruments or not actions:
        return instruments, actions

    pruned_actions = {
        a for a in actions
        if any(a in _INSTRUMENT_ALLOWED_ACTIONS.get(i, set()) for i in instruments)
    }
    if not pruned_actions:
        pruned_actions = set(actions)

    pruned_instr = {
        i for i in instruments
        if _INSTRUMENT_ALLOWED_ACTIONS.get(i, set()) & pruned_actions
    }
    if not pruned_instr:
        pruned_instr = set(instruments)

    ctx = step_context.lower()
    for step_label, c in _STEP_INSTRUMENT_CONSTRAINTS.items():
        if step_label in ctx:
            excl = c.get("must_exclude", set())
            incl = c.get("must_include", set())
            removable = pruned_instr & excl
            if removable and len(pruned_instr) - len(removable) >= 1:
                pruned_instr -= excl
            pruned_instr |= (incl & INSTRUMENT_LABELS)

    return pruned_instr, pruned_actions


def apply_action_cooccurrence_constraints(actions):
    """Remove actions forming an impossible co-occurrence pair (Fig C.13)."""
    if len(actions) < 2:
        return actions
    priority = ["cut","cauterize","push","pull","hold","open","close",
                "grasp","release","travel","suction","open something","other"]
    pruned = set(actions)
    changed = True
    while changed:
        changed = False
        for pair in _INCOMPATIBLE_ACTION_PAIRS:
            if pair <= pruned:
                a, b = tuple(pair)
                if "still" in pair:
                    pruned.discard("still")
                else:
                    ra = priority.index(a) if a in priority else 99
                    rb = priority.index(b) if b in priority else 99
                    to_drop = b if ra <= rb else a
                    if len(pruned) > 1:
                        pruned.discard(to_drop)
                changed = True
                break
    return pruned if pruned else actions


def apply_phase_step_hierarchy(predicted_step, predicted_phase):
    """Return 'idle' if step is inconsistent with predicted phase (medium confidence)."""
    if not predicted_phase or not predicted_step:
        return predicted_step
    if predicted_phase == "idle" or predicted_step == "idle":
        return predicted_step
    valid = _PHASE_TO_VALID_STEPS.get(predicted_phase, set())
    if valid and predicted_step not in valid:
        return "idle"
    return predicted_step

# ─────────────────────────────────────────────────────────────────────────────
# 6.  File Loading
# ─────────────────────────────────────────────────────────────────────────────

def _sniff_jsonl(fp):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s: return s.startswith("{")
    return False

def load_predictions_tsv(fp):
    samples = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) >= 2:
                samples.append((parts[0].strip(), parts[1].strip(), ""))
    return samples

def load_predictions_jsonl(fp):
    samples = []
    with open(fp, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if "predict" not in obj or "label" not in obj:
                raise KeyError(f"Line {n}: missing predict/label")
            samples.append((str(obj["predict"]), str(obj["label"]),
                             str(obj.get("prompt", ""))))
    return samples

def load_predictions(fp):
    return (load_predictions_jsonl(fp)
            if fp.lower().endswith(".jsonl") or _sniff_jsonl(fp)
            else load_predictions_tsv(fp))

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Text Helpers
# ─────────────────────────────────────────────────────────────────────────────

_THINK_END = ("</think>", "<|im_end|>")

def extract_answer_text(pred):
    t = pred.strip()
    for m in _THINK_END:
        if m in t: t = t.split(m)[-1]
    return t.strip()

def _token_recall(cand, hay_tok):
    ct = set(re.findall(r"[a-z0-9]+", cand.lower()))
    return len(ct & hay_tok) / len(ct) if ct else 0.0

def _phrase_in(phrase, hay):
    if phrase not in hay: return False
    if len(phrase) <= 6:
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", hay))
    return True

def _last_sentence(answer):
    t = answer.strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2: return parts[-1]
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    return lines[-1] if lines else t

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Task Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_task(label, prompt=""):
    pl = prompt.lower()
    for name, snip in _TASK_PROMPT_SNIPPETS:
        if snip in pl: return name
    l = label.strip().lower()
    if "," in l or ";" in l:
        parts = [p.strip() for p in re.split(r"[,;]", l) if p.strip()]
        if parts and all(p in INSTRUMENT_LABELS for p in parts): return "instrument"
        if parts and all(p in ACTION_LABELS for p in parts):     return "action"
    if l in STEP_LABELS:       return "step"
    if l in PHASE_LABELS:      return "phase"
    if l in INSTRUMENT_LABELS: return "instrument"
    if l in ACTION_LABELS:     return "action"
    return "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _apply_step_synonyms(text):
    t = " ".join(text.lower().split())
    for phrase, label in sorted(_STEP_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if phrase in t: return label
    return None

def normalize_single_label(pred, task, predicted_context=""):
    cands  = TASK_TO_LABELS[task]
    answer = extract_answer_text(pred)
    hay    = " ".join(answer.lower().split())
    hay_t  = set(re.findall(r"[a-z0-9]+", hay))

    if task == "step":
        syn = _apply_step_synonyms(hay)
        if syn and syn in STEP_LABELS: return syn

    best, blen = None, -1
    for c in cands:
        if _phrase_in(c, hay) and len(c) > blen:
            best, blen = c, len(c)

    if task == "step" and best and best in PHASE_LABELS and best not in STEP_LABELS:
        step_only = STEP_LABELS - {best}
        best2, blen2 = None, -1
        for c in step_only:
            if _phrase_in(c, hay) and len(c) > blen2:
                best2, blen2 = c, len(c)
        if best2: return best2
        ls = " ".join(_last_sentence(answer).lower().split())
        syn = _apply_step_synonyms(ls)
        if syn and syn in STEP_LABELS: return syn
    elif best:
        return best

    ls = " ".join(_last_sentence(answer).lower().split())
    if ls:
        best2, blen2 = None, -1
        for c in cands:
            if _phrase_in(c, ls) and len(c) > blen2:
                best2, blen2 = c, len(c)
        if best2: return best2
        close = difflib.get_close_matches(ls, list(cands), n=1, cutoff=0.55)
        if close: return close[0]

    scored = sorted([(_token_recall(c, hay_t), c) for c in cands],
                    key=lambda x: (-x[0], -len(x[1])))
    if scored[0][0] >= 0.72: return scored[0][1]

    llines = [l.strip() for l in answer.split("\n") if len(l.strip()) > 12]
    if llines:
        close = difflib.get_close_matches(llines[-1].lower(), list(cands), n=1, cutoff=0.55)
        if close: return close[0]

    return scored[0][1]

def parse_gt_multilabel(label_raw):
    parts = re.split(r"[,;]", label_raw)
    return frozenset(p.strip().lower() for p in parts if p.strip())

_IDLE_INSTR_PHRASES = [
    "idle, with no active", "not actively manipulating",
    "positioned but not", "no instrument", "no active manipulation",
    "instruments are idle", "no visible instrument",
]

def _expand_instrument_synonyms(hay, act_tok, ctx=""):
    found = set()
    h = " ".join(hay.lower().split())
    if any(p in h for p in _IDLE_INSTR_PHRASES):
        return found
    _AMB = ["energy device", "electrosurgical device", "electrosurgical instrument"]
    if any(t in h for t in _AMB):
        r = resolve_energy_device(h, act_tok, ctx)
        if r: found.add(r)
    for phrase, label in sorted(_INSTRUMENT_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if _phrase_in(phrase, h):
            found.add(label)
    return found

def _expand_action_synonyms(hay):
    found = set()
    h = " ".join(hay.lower().split())
    still_p = ["idle","no active","positioned but not","not actively",
               "not manipulating","no instrument-tissue"]
    if any(p in h for p in still_p):
        found.add("still")
    for phrase, label in sorted(_ACTION_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if _phrase_in(phrase, h):
            found.add(label)
    return found

def normalize_multilabel(pred, task, predicted_context=""):
    """
    Full normalisation pipeline:
      STEP 1: synonym / phrase expansion  (additive)
      STEP 2: co-occurrence constraint pruning  (subtractive)

    Synonym mapping ALWAYS runs before constraints.
    Constraints NEVER reassign — they only remove impossible labels.
    """
    cands  = TASK_TO_LABELS[task]
    answer = extract_answer_text(pred)
    hay    = " ".join(answer.lower().split())
    hay_t  = set(re.findall(r"[a-z0-9]+", hay))

    # ── STEP 1: Synonym Expansion ─────────────────────────────────────────────
    matched = {c for c in cands if _phrase_in(c, hay)}
    for c in cands:
        if c not in matched and _token_recall(c, hay_t) >= 0.60:
            matched.add(c)
    if task == "instrument":
        act_tok = set(re.findall(r"[a-z]+", hay))
        matched |= _expand_instrument_synonyms(hay, act_tok, predicted_context)
    elif task == "action":
        matched |= _expand_action_synonyms(hay)
    matched = matched & cands

    if not matched:
        scored = sorted([(_token_recall(c, hay_t), c) for c in cands],
                        key=lambda x: (-x[0], -len(x[1])))
        matched = {scored[0][1]}

    # ── STEP 2: Constraint Pruning ────────────────────────────────────────────
    if task == "instrument":
        companion = (_expand_action_synonyms(hay) & ACTION_LABELS) or \
                    {c for c in ACTION_LABELS if _phrase_in(c, hay)}
        p_instr, _ = apply_instrument_action_constraints(
            matched, companion, step_context=predicted_context)
        matched = p_instr if p_instr else matched

    elif task == "action":
        # Detect instrument names from the answer text to enable instrument-aware pruning
        detected_instr = {c for c in INSTRUMENT_LABELS if _phrase_in(c, hay)}
        detected_instr |= (
            _expand_instrument_synonyms(hay, set(re.findall(r"[a-z]+", hay)),
                                         predicted_context) & INSTRUMENT_LABELS
        )
        # If instruments are identifiable, prune impossible actions for those instruments
        if detected_instr:
            _, matched = apply_instrument_action_constraints(
                detected_instr, matched, step_context=predicted_context)
        # Always apply action co-occurrence constraint
        matched = apply_action_cooccurrence_constraints(matched)

    return frozenset(matched)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_single(preds, labels, task_name):
    print(f"\n{'='*62}")
    print(f" Task: {task_name.upper()} ({len(labels)} samples) [single-label]")
    print(f"{'='*62}")
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Macro F1   : {mf1*100:.2f}%")
    lb = LabelBinarizer()
    yb = lb.fit_transform(labels)
    ys = np.zeros_like(yb, dtype=float)
    pi = {c: i for i, c in enumerate(lb.classes_)}
    for i, p in enumerate(preds):
        if p in pi: ys[i, pi[p]] = 1.0
    if yb.shape[1] == 1:
        mAP = float("nan")
        print("  mAP        : N/A")
    else:
        aps = [average_precision_score(yb[:, c], ys[:, c])
               for c in range(yb.shape[1]) if yb[:, c].sum() > 0]
        mAP = float(np.mean(aps)) if aps else float("nan")
        print(f"  mAP        : {mAP*100:.2f}%")
    print(f"\n  Per-class F1:")
    for line in classification_report(labels, preds, zero_division=0).split("\n"):
        print("  " + line)
    return {"task": task_name, "n_samples": len(labels), "accuracy": acc,
            "macro_f1": mf1, "mAP": mAP, "micro_f1": None, "exact_match": None}

def compute_metrics_multilabel(preds_sets, labels_sets, task_name):
    print(f"\n{'='*62}")
    print(f" Task: {task_name.upper()} ({len(labels_sets)} samples) [multi-label]")
    print(f"{'='*62}")
    classes = sorted(TASK_TO_LABELS[task_name])
    mlb = MultiLabelBinarizer(classes=classes)
    yt  = mlb.fit_transform([list(s) for s in labels_sets])
    yp  = mlb.transform([list(s) for s in preds_sets])
    em  = float(np.mean(np.all(yt == yp, axis=1)))
    mf1 = f1_score(yt, yp, average="macro", zero_division=0)
    uf1 = f1_score(yt, yp, average="micro", zero_division=0)
    print(f"  Exact Match : {em*100:.2f}%")
    print(f"  Macro F1    : {mf1*100:.2f}%")
    print(f"  Micro F1    : {uf1*100:.2f}%")
    try:
        mAP = float(average_precision_score(yt, yp, average="macro"))
    except ValueError:
        mAP = float("nan")
    print(f"  mAP (macro) : {mAP*100:.2f}%" if not np.isnan(mAP) else "  mAP : N/A")
    print(f"\n  Per-label F1:")
    for line in classification_report(yt, yp, target_names=classes, zero_division=0).split("\n"):
        print("  " + line)
    return {"task": task_name, "n_samples": len(labels_sets), "accuracy": None,
            "macro_f1": mf1, "micro_f1": uf1, "exact_match": em, "mAP": mAP}

# ─────────────────────────────────────────────────────────────────────────────
# 11. Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test():
    tests = [
        # Synonym expansion (v3)
        {"task":"step",   "note":"spelling synonym",
         "predict":"The step is prevesical dissection.", "label":"prevessical dissection"},
        {"task":"step",   "note":"step synonym",
         "predict":"Passing a suture through the bladder neck.",
         "label":"pass suture to the bladder neck"},
        {"task":"action", "note":"retraction -> pull",
         "predict":"The scissors is retracting tissue away.", "label":"pull"},
        {"task":"action", "note":"coagulation -> cauterize",
         "predict":"The forceps is coagulating the vessel.", "label":"cauterize"},
        # Energy device resolution
        {"task":"instrument", "note":"energy device + cut -> MCS",
         "predict":"An energy device is cutting and dissecting tissue.",
         "label":"monopolar curved scissors"},
        {"task":"instrument", "note":"energy device + cauterize only -> BF",
         "predict":"An energy device is coagulating a vessel.",
         "label":"bipolar forceps"},
        # Instrument x Action constraint (v4)
        {"task":"action", "note":"cauterize pruned — PF cannot cauterize",
         "predict":"The prograsp forceps is cauterizing the tissue.",
         "label":"hold"},
        {"task":"action", "note":"cut pruned — SI cannot cut",
         "predict":"The suction instrument is suctioning and cutting.",
         "label":"suction"},
        # Action co-occurrence constraint (v4)
        {"task":"action", "note":"still pruned — incompatible with cut",
         "predict":"The instrument is still and cutting tissue.", "label":"cut"},
        # Step instrument constraint (v4)
        {"task":"instrument", "note":"MCS excluded during suture step",
         "context":"what surgical instruments are present during pass suture to the bladder neck",
         "predict":"monopolar curved scissors and large needle driver present",
         "label":"large needle driver"},
        # Idle suppression
        {"task":"action", "note":"idle -> still",
         "predict":"The instruments are idle, with no active manipulation.", "label":"still"},
    ]

    print("\n=== Self-test (v4 — constraint rules) ===\n")
    passes, fails = 0, 0
    for t in tests:
        task = t["task"]
        ctx  = t.get("context", "")
        note = t.get("note", "")
        if task in MULTILABEL_TASKS:
            gt = parse_gt_multilabel(t["label"])
            pr = normalize_multilabel(t["predict"], task, predicted_context=ctx)
            ok = gt <= pr   # GT labels are all present in prediction
            status = "OK" if ok else f"PARTIAL {len(gt & pr)}/{len(gt)}"
        else:
            gt = t["label"].strip().lower()
            pr = normalize_single_label(t["predict"], task)
            ok = pr == gt
            status = "OK" if ok else "WRONG"
        passes += ok; fails += not ok
        print(f"  [{task:<10}] {status:<14}  GT: {t['label']}")
        if not ok:
            print(f"  {' '*12}              GOT: {pr if isinstance(pr, str) else sorted(pr)}")
        print(f"  {' '*12}              {note}")
        print()
    print(f"Result: {passes}/{passes+fails} passed\n")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="GraSP eval metrics v4")
    p.add_argument("predictions_file", nargs="?")
    p.add_argument("--output_json", default="eval_results.json")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _run_self_test(); return

    if not args.predictions_file:
        p.error("predictions_file required unless --self-test")

    print(f"\nLoading: {args.predictions_file}")
    samples = load_predictions(args.predictions_file)
    print(f"Samples : {len(samples)}")

    td  = defaultdict(lambda: {"preds": [], "labels": [], "multilabel": False})
    unk = 0
    for pred, label, prompt in samples:
        task = detect_task(label, prompt=prompt)
        if task == "unknown":
            unk += 1; continue
        if task in MULTILABEL_TASKS:
            td[task]["multilabel"] = True
            td[task]["preds"].append(
                normalize_multilabel(pred, task, predicted_context=prompt))
            td[task]["labels"].append(parse_gt_multilabel(label))
        else:
            td[task]["preds"].append(
                normalize_single_label(pred, task, predicted_context=prompt))
            td[task]["labels"].append(label.strip().lower())

    if unk:
        print(f"\nWarning: {unk} samples skipped (task not recognised).")

    results = []
    for name in ["phase", "step", "instrument", "action"]:
        d = td.get(name)
        if not d or not d["labels"]:
            print(f"\nNo samples for '{name}' — skipped."); continue
        results.append(
            compute_metrics_multilabel(d["preds"], d["labels"], name)
            if d["multilabel"] else
            compute_metrics_single(d["preds"], d["labels"], name)
        )

    print(f"\n{'='*70}\n SUMMARY\n{'='*70}")
    print(f"  {'Task':<14} {'N':>7} {'Acc/Exact':>16} {'MacroF1':>9} {'mAP':>8}")
    print(f"  {'-'*56}")
    for r in results:
        mAP_s = (f"{r['mAP']*100:.2f}%"
                 if r["mAP"] and not np.isnan(r["mAP"]) else "N/A")
        acc_s = (f"{r['exact_match']*100:.2f}% (EM)"
                 if r["exact_match"] is not None else f"{r['accuracy']*100:.2f}%")
        print(f"  {r['task']:<14} {r['n_samples']:>7} {acc_s:>16} "
              f"{r['macro_f1']*100:>8.2f}% {mAP_s:>8}")

    def _safe(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    with open(args.output_json, "w") as f:
        json.dump([{k: _safe(v) if isinstance(v, float) else v
                    for k, v in r.items()} for r in results], f, indent=2)
    print(f"\nSaved: {args.output_json}\n")

if __name__ == "__main__":
    main()
