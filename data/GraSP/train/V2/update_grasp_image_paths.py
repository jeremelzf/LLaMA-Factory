#!/usr/bin/env python3
"""
Update GraSP JSONL entries:
1) replace old "image" paths with new base paths,
2) normalize "image" as a one-item list: ["..."],
3) prepend "<image>\\n" to the human turn when missing.

Reads and writes UTF-8 JSONL line-by-line with progress and summary stats.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# === Easy-to-change base path ===
# Both old path patterns are mapped to this same base directory.
BASE_FRAMES = "/scratch/e0957602/BN4101/frames/1fps_frames"

# Old path markers to detect
MARKER_30FPS = "/GraSP/train/frames/"
MARKER_1FPS = "/GraSP/GraSP_1fps/frames/"


def _map_image_path(old_path: str) -> tuple[str | None, str]:
    """
    Returns (new_path_or_None, kind) where kind in {"type1","type2","unknown"}.
    Both types are mapped to the same BASE_FRAMES.
    """
    if MARKER_30FPS in old_path:
        tail = old_path.split(MARKER_30FPS, 1)[1].lstrip("/")
        return f"{BASE_FRAMES}/{tail}", "type1"
    if MARKER_1FPS in old_path:
        tail = old_path.split(MARKER_1FPS, 1)[1].lstrip("/")
        return f"{BASE_FRAMES}/{tail}", "type2"
    return None, "unknown"


def update_jsonl_image_paths(input_path: Path, output_path: Path) -> None:
    total = 0
    count_type1 = 0
    count_type2 = 0
    count_unknown = 0
    count_human_prefixed = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}") from e

            image_value = obj.get("image")
            if isinstance(image_value, str):
                old_image = image_value
            elif isinstance(image_value, list) and len(image_value) == 1 and isinstance(image_value[0], str):
                old_image = image_value[0]
            else:
                print(
                    f"WARNING line {line_no}: unsupported 'image' field ({image_value!r}); keeping line unchanged",
                    file=sys.stderr,
                )
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            new_image, kind = _map_image_path(old_image)
            if kind == "unknown" or new_image is None:
                count_unknown += 1
                print(f"WARNING line {line_no}: unknown image path pattern: {old_image}", file=sys.stderr)
                final_image = old_image
            else:
                if kind == "type1":
                    count_type1 += 1
                elif kind == "type2":
                    count_type2 += 1
                final_image = new_image

            # Normalize image to list form.
            obj["image"] = [final_image]

            # Prepend "<image>\\n" to human turn if missing.
            conversations = obj.get("conversations")
            if isinstance(conversations, list):
                for turn in conversations:
                    if not isinstance(turn, dict):
                        continue
                    if turn.get("from") != "human":
                        continue
                    value = turn.get("value")
                    if isinstance(value, str) and not value.startswith("<image>"):
                        turn["value"] = "<image>\n" + value
                        count_human_prefixed += 1
                    break

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if total % 10_000 == 0:
                print(f"Processed {total} lines...", file=sys.stderr)

    print("=== Summary ===", file=sys.stderr)
    print(f"Total entries: {total}", file=sys.stderr)
    print(f"Type1 updated: {count_type1}", file=sys.stderr)
    print(f"Type2 updated: {count_type2}", file=sys.stderr)
    print(f"Unknown/unchanged: {count_unknown}", file=sys.stderr)
    print(f"Human turns prefixed: {count_human_prefixed}", file=sys.stderr)


def main() -> None:
    """
    Usage:
      python update_grasp_image_paths.py [input.jsonl] [output.jsonl]

    Defaults to files in the same directory as this script.
    """
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "GraSP_PhaseStep_Recognition.jsonl"
    default_output = script_dir / "GraSP_PhaseStep_Recognition_final.jsonl"

    input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else default_input
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else default_output

    update_jsonl_image_paths(input_path, output_path)


if __name__ == "__main__":
    main()

