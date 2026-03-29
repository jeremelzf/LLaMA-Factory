#!/usr/bin/env python3
"""
Update GraSP JSONL entries:
1) replace old "image" paths with new base paths,
2) zero-pad 30fps frame filenames to 9 digits under BASE_30FPS,
3) normalize "image" as a one-item list: ["..."],
4) prepend "<image>\\n" to the human turn when missing.

Reads and writes UTF-8 JSONL line-by-line with progress and summary stats.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# === Easy-to-change base paths ===
BASE_30FPS = "/scratch/e0957602/BN4101/frames/30fps_frames"
BASE_1FPS = "/scratch/e0957602/BN4101/frames/1fps_frames"

# Old path markers to detect
MARKER_30FPS = "/GraSP/train/frames/"
MARKER_1FPS = "/GraSP/GraSP_1fps/frames/"

FRAME_PAD_WIDTH = 9


def _pad_30fps_frame_filename(path: str) -> str:
    """
    For paths under BASE_30FPS, rewrite the last path segment (e.g. 04762.jpg) so the
    numeric frame stem is zero-padded to FRAME_PAD_WIDTH digits.
    """
    prefix = BASE_30FPS + "/"
    if not path.startswith(prefix):
        return path
    rel = path[len(prefix) :].lstrip("/")
    parts = rel.split("/")
    if len(parts) < 2:
        return path
    fname = parts[-1]
    stem = Path(fname).stem
    suffix = Path(fname).suffix
    if not stem.isdigit():
        return path
    padded_name = f"{int(stem):0{FRAME_PAD_WIDTH}d}{suffix}"
    case_parts = parts[:-1]
    return f"{BASE_30FPS}/{'/'.join(case_parts)}/{padded_name}"


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
                if kind == "30fps":
                    count_30fps += 1
                elif kind == "1fps":
                    count_1fps += 1
                final_image = new_image

            final_image = _pad_30fps_frame_filename(final_image)

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
    print(f"30fps updated: {count_30fps}", file=sys.stderr)
    print(f"1fps updated: {count_1fps}", file=sys.stderr)
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
    default_output = script_dir / "GraSP_PhaseStep_Recognition_updated.jsonl"

    input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else default_input
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else default_output

    update_jsonl_image_paths(input_path, output_path)


if __name__ == "__main__":
    main()

