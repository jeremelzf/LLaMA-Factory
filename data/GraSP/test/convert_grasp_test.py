import json
import os
import random
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
LONG_TERM_JSON  = "data/GraSP/test/grasp_long-term_test.json"
SHORT_TERM_JSON = "data/GraSP/test/grasp_short-term_test.json"
FRAMES_DIR      = "/scratch/e0957602/BN4101/frames/1fps_frames"
OUTPUT_JSONL    = "data/GraSP/test/GraSP_test.jsonl"
# ─────────────────────────────────────────────────────────────────────────────

# ── LABEL MAPS ───────────────────────────────────────────────────────────────
PHASE_NAMES = {
    0: "Idle",
    1: "Left pelvic isolated lymphadenectomy",
    2: "Right pelvic isolated lymphadenectomy",
    3: "Developing the Space of Retzius",
    4: "Ligation of the deep dorsal venous complex",
    5: "Bladder neck identification and transection",
    6: "Seminal vesicle dissection",
    7: "Development of the plane between the prostate and rectum",
    8: "Prostatic pedicle control",
    9: "Severing of the prostate from the urethra",
    10: "Bladder neck reconstruction",
}

STEP_NAMES = {
    0: "Idle",
    1: "Identification and dissection of the Iliac vein and artery",
    2: "Cutting and dissection of the external iliac veins lymph node",
    3: "Obturator nerve and vessel path identification, dissection and cutting of the obturator lymph nodes",
    4: "Insert the lymph nodes in retrieval bags",
    5: "Prevessical dissection",
    6: "Ligation of the dorsal venous complex",
    7: "Prostate dissection until the levator ani",
    8: "Seminal vesicle dissection",
    9: "Dissection of Denonviliers fascia",
    10: "Cut the tissue between the prostate and the urethra",
    11: "Hold prostate",
    12: "Insert prostate in retrieval bag",
    13: "Pass suture to the urethra",
    14: "Pass suture to the bladder neck",
    15: "Pull suture",
    16: "Tie suture",
    17: "Suction",
    18: "Cut suture or tissue",
    19: "Cut between the prostate and bladder neck",
    20: "Vascular pedicle control",
}

INSTRUMENT_NAMES = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Monopolar Curved Scissors",
    5: "Suction Instrument",
    6: "Clip Applier",
    7: "Laparoscopic Grasper",
}

ACTION_NAMES = {
    1: "Cauterize",   2: "Close",         3: "Cut",
    4: "Grasp",       5: "Hold",          6: "Open",
    7: "Open Something", 8: "Pull",       9: "Push",
    10: "Release",    11: "Still",        12: "Suction",
    13: "Travel",     14: "Other",
}

# ── FIXED QUESTIONS (one per task) ───────────────────────────────────────────
Q_PHASE      = "What surgical phase is currently being performed in this prostatectomy image?"
Q_STEP       = "What surgical step is currently being performed in this prostatectomy image?"
Q_INSTRUMENT = "What surgical instruments are visible in this prostatectomy image?"
Q_ACTION     = "What atomic actions are being performed by the surgical instruments in this image?"

# ── HELPERS ──────────────────────────────────────────────────────────────────
def make_entry(entry_id, image_path, question, answer, keywords):
    """Create one ShareGPT-format JSONL entry."""
    return {
        "id": entry_id,
        "image": [image_path],
        "keywords": keywords,
        "conversations": [
            {"from": "human",  "value": f"<image>\n{question}"},
            {"from": "gpt",    "value": answer, "original_value": answer},
        ],
        "level": 3,
    }

def _format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # ---------- Load long-term annotations (phase + step, all 1fps) ----------
    print("Loading long-term annotations...")
    with open(LONG_TERM_JSON) as f:
        lt = json.load(f)

    lt_id_to_filename = {img["id"]: img["file_name"] for img in lt["images"]}

    # Map image_id → (phase_id, step_id) from long-term annotations
    lt_labels = {}
    for ann in lt["annotations"]:
        lt_labels[ann["image_id"]] = {
            "phase": ann["phases"],
            "step":  ann["steps"],
        }

    # ---------- Load short-term annotations (all 4 tasks, 35s keyframes) -----
    print("Loading short-term annotations...")
    with open(SHORT_TERM_JSON) as f:
        st = json.load(f)

    st_id_to_filename = {img["id"]: img["file_name"] for img in st["images"]}

    # Group short-term annotations by image_id
    # Each image can have multiple instrument instances → collect all
    st_by_image = {}
    for ann in st["annotations"]:
        iid = ann["image_id"]
        if iid not in st_by_image:
            st_by_image[iid] = {
                "phase":       ann["phases"],
                "step":        ann["steps"],
                "instruments": set(),
                "actions":     set(),
            }
        # Handle both "category_id" and "instruments" field names
        instrument_id = ann.get("category_id") or ann.get("instruments")
        if instrument_id:
            st_by_image[iid]["instruments"].add(instrument_id)

        for action_id in ann.get("actions", []):
            st_by_image[iid]["actions"].add(action_id)

    # Build set of short-term filenames for quick lookup
    st_filenames = set(st_id_to_filename.values())

    # ---------- Generate JSONL entries ----------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    entries = []
    skipped = 0

    # --- Long-term: phase-stratified 23% random sample (phase + step) ---
    print("Bucketing long-term frames by phase (existence-checked)...")
    lt_phase_buckets = defaultdict(list)  # phase_id -> [image_id, ...] (only existing on disk)
    lt_existing_total_by_phase = defaultdict(int)
    for image_id, labels in lt_labels.items():
        file_name = lt_id_to_filename[image_id]  # e.g. CASE041/00000.jpg
        image_path = os.path.join(FRAMES_DIR, file_name)
        if not os.path.exists(image_path):
            skipped += 1
            continue
        phase_id = labels["phase"]
        lt_phase_buckets[phase_id].append(image_id)
        lt_existing_total_by_phase[phase_id] += 1

    rng = random.Random(42)
    sampled_lt_image_ids = set()
    lt_sampled_by_phase = {}

    print("Sampling 23% per phase (min 1 if phase has any frames)...")
    for phase_id in range(0, 11):
        bucket = lt_phase_buckets.get(phase_id, [])
        n = len(bucket)
        if n == 0:
            k = 0
            sampled = []
        else:
            k = max(1, int(round(n * 0.23)))
            k = min(k, n)
            sampled = rng.sample(bucket, k)
        sampled_lt_image_ids.update(sampled)
        lt_sampled_by_phase[phase_id] = (n, len(sampled))

    print("\n=== Long-term stratified sampling breakdown (existing-on-disk only) ===")
    for phase_id in range(0, 11):
        total_n, sampled_k = lt_sampled_by_phase.get(phase_id, (0, 0))
        phase_name = PHASE_NAMES.get(phase_id, f"Phase_{phase_id}")
        print(f"Phase {phase_id:>2} ({phase_name}): {total_n} -> {sampled_k}")

    print("\nGenerating long-term (phase + step) entries from sampled frames...")
    for image_id in sorted(sampled_lt_image_ids):
        labels = lt_labels[image_id]
        file_name = lt_id_to_filename[image_id]
        image_path = os.path.join(FRAMES_DIR, file_name)

        phase_name = PHASE_NAMES.get(labels["phase"], f"Phase_{labels['phase']}")
        step_name  = STEP_NAMES.get(labels["step"],   f"Step_{labels['step']}")

        entries.append(make_entry(
            entry_id   = f"lt_{image_id}_phase",
            image_path = image_path,
            question   = Q_PHASE,
            answer     = phase_name,
            keywords   = ["Phase&Step Recognition"],
        ))

        entries.append(make_entry(
            entry_id   = f"lt_{image_id}_step",
            image_path = image_path,
            question   = Q_STEP,
            answer     = step_name,
            keywords   = ["Phase&Step Recognition"],
        ))

    # --- Short-term: instrument + action entries for 35s keyframes only ---
    print("Generating short-term (instrument + action) entries...")
    for image_id, labels in st_by_image.items():
        file_name  = st_id_to_filename[image_id]
        image_path = os.path.join(FRAMES_DIR, file_name)

        if not os.path.exists(image_path):
            skipped += 1
            continue

        # Instrument answer: comma-separated list of unique instrument names
        instrument_names = sorted([
            INSTRUMENT_NAMES.get(i, f"Instrument_{i}")
            for i in labels["instruments"]
        ])
        instrument_answer = ", ".join(instrument_names)

        # Action answer: comma-separated list of unique action names
        action_names = sorted([
            ACTION_NAMES.get(a, f"Action_{a}")
            for a in labels["actions"]
        ])
        action_answer = ", ".join(action_names)

        # Instrument question entry
        entries.append(make_entry(
            entry_id   = f"st_{image_id}_instrument",
            image_path = image_path,
            question   = Q_INSTRUMENT,
            answer     = instrument_answer,
            keywords   = ["Instrument&Action Recognition"],
        ))

        # Action question entry
        entries.append(make_entry(
            entry_id   = f"st_{image_id}_action",
            image_path = image_path,
            question   = Q_ACTION,
            answer     = action_answer,
            keywords   = ["Instrument&Action Recognition"],
        ))

    # ---------- Write output --------------------------------------------------
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n=== Summary ===")
    print(f"Total entries written : {len(entries)}")
    print(f"  Phase entries       : {sum(1 for e in entries if 'phase' in e['id'])}")
    print(f"  Step entries        : {sum(1 for e in entries if 'step' in e['id'])}")
    print(f"  Instrument entries  : {sum(1 for e in entries if 'instrument' in e['id'])}")
    print(f"  Action entries      : {sum(1 for e in entries if 'action' in e['id'])}")
    print(f"Frames skipped        : {skipped}")
    print(f"Output written to     : {OUTPUT_JSONL}")

    est_batch1 = len(entries) * 20
    est_batch4 = len(entries) * 5
    print("Estimated eval time   :")
    print(f"  batch=1 (~20s/entry): {_format_seconds(est_batch1)}")
    print(f"  batch=4 (~5s/entry) : {_format_seconds(est_batch4)}")


if __name__ == "__main__":
    main()