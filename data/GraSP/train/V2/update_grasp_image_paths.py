#!/usr/bin/env python3
"""
Update only the top-level "image" field paths in a JSONL file.

Reads the input JSONL line-by-line (UTF-8), replaces old server paths found in the
"image" field, and writes updated lines to a new JSONL file (UTF-8).

All other fields and formatting in each original JSON line are preserved as-is,
except for the string value of the "image" field.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# === Easy-to-change base paths ===
BASE_30FPS = "/scratch/e0957602/BN4101/frames/30fps_frames"
BASE_1FPS = "/scratch/e0957602/BN4101/frames/1fps_frames"

# Old path markers to detect
MARKER_30FPS = "/GraSP/train/frames/"
MARKER_1FPS = "/GraSP/GraSP_1fps/frames/"


# Match:  "image"  :  "<json string>"
# (robust to escapes inside the JSON string)
IMAGE_FIELD_RE = re.compile(r'("image"\s*:\s*)"(?:\\.|[^"\\])*"')


def _map_image_path(old_path: str) -> tuple[str | None, str]:
    """
    Returns (new_path_or_None, kind) where kind in {"30fps","1fps","unknown"}.
    """
    if MARKER_30FPS in old_path:
        tail = old_path.split(MARKER_30FPS, 1)[1].lstrip("/")
        return f"{BASE_30FPS}/{tail}", "30fps"
    if MARKER_1FPS in old_path:
        tail = old_path.split(MARKER_1FPS, 1)[1].lstrip("/")
        return f"{BASE_1FPS}/{tail}", "1fps"
    return None, "unknown"


def update_jsonl_image_paths(input_path: Path, output_path: Path) -> None:
    total = 0
    count_30fps = 0
    count_1fps = 0
    count_unknown = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            total += 1

            # Preserve exact original newlines; operate on content only.
            has_newline = line.endswith("\n")
            raw = line[:-1] if has_newline else line

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}") from e

            old_image = obj.get("image")
            if not isinstance(old_image, str):
                print(
                    f"WARNING line {line_no}: missing/non-string 'image' field ({old_image!r}); keeping line unchanged",
                    file=sys.stderr,
                )
                fout.write(line)
                continue

            new_image, kind = _map_image_path(old_image)
            if kind == "unknown" or new_image is None:
                count_unknown += 1
                print(f"WARNING line {line_no}: unknown image path pattern: {old_image}", file=sys.stderr)
                fout.write(line)
            else:
                if kind == "30fps":
                    count_30fps += 1
                elif kind == "1fps":
                    count_1fps += 1

                # Replace only the JSON string value inside the top-level "image" field,
                # while keeping all other characters in the line identical.
                replacement = r"\1" + json.dumps(new_image, ensure_ascii=False)
                updated, nsubs = IMAGE_FIELD_RE.subn(replacement, raw, count=1)

                if nsubs != 1:
                    # If we can't safely locate the "image" field in the raw line, don't rewrite it.
                    count_unknown += 1
                    print(
                        f"WARNING line {line_no}: couldn't locate 'image' field text to replace; keeping line unchanged. image={old_image}",
                        file=sys.stderr,
                    )
                    fout.write(line)
                else:
                    fout.write(updated + ("\n" if has_newline else ""))

            if total % 10_000 == 0:
                print(f"Processed {total} lines...", file=sys.stderr)

    print("=== Summary ===", file=sys.stderr)
    print(f"Total entries: {total}", file=sys.stderr)
    print(f"30fps updated: {count_30fps}", file=sys.stderr)
    print(f"1fps updated: {count_1fps}", file=sys.stderr)
    print(f"Unknown/unchanged: {count_unknown}", file=sys.stderr)


def main() -> None:
    """
    Usage:
      python update_grasp_image_paths.py [input.jsonl] [output.jsonl]

    Defaults to files in the same directory as this script.
    """
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "GraSP_PhaseStep_Recognition.jsonl"
    default_output = script_dir / "GraSP_PhaseStep_Recognition_updated.jsonl"

    input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else default_input
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else default_output

    update_jsonl_image_paths(input_path, output_path)


if __name__ == "__main__":
    main()

