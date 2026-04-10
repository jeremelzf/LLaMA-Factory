import json
import argparse

DEFAULT_INPUT_PATH = "data/GraSP/test/GraSP_test_blob.jsonl"
DEFAULT_OUTPUT_PATH = "data/GraSP/test/GraSP_test.jsonl"


def _print_first_separator_bytes(data: bytes) -> None:
    """
    Print raw bytes around the first detected object boundary.

    We look for the common marker `"level": 3}` (end of an object in this dataset),
    then show bytes immediately following it to confirm the exact separator.
    """
    needle = b"\"level\": 3}"
    idx = data.find(needle)
    if idx == -1:
        first_brace = data.find(b"}")
        if first_brace == -1:
            print("Could not find any '}' byte in the file preview.")
            return
        start = max(0, first_brace - 40)
        end = min(len(data), first_brace + 80)
        snippet = data[start:end]
        print("First '}' bytes snippet (repr):", repr(snippet))
        print("First '}' bytes snippet (hex) :", snippet.hex(" ", 1))
        return

    end_idx = idx + len(needle)
    start = max(0, end_idx - 40)
    end = min(len(data), end_idx + 80)
    snippet = data[start:end]
    after = data[end_idx : min(len(data), end_idx + 30)]

    print('First `"level": 3}` boundary snippet (repr):', repr(snippet))
    print('Bytes immediately after boundary (repr):', repr(after))
    print('Bytes immediately after boundary (hex) :', after.hex(" ", 1))


def _skip_separators(s: str, pos: int) -> int:
    """
    Skip whitespace and common separators between concatenated JSON objects.

    IMPORTANT: We intentionally do NOT split on '\\n' globally because this dataset
    contains fields like '<image>\\nQuestion...' inside JSON strings.
    """
    n = len(s)
    while pos < n:
        ch = s[pos]
        if ch.isspace():
            pos += 1
            continue
        # Literal backslash-escaped newlines between objects: "\\n" or "\\r\\n"
        if s.startswith("\\r\\n", pos):
            pos += 4
            continue
        if s.startswith("\\n", pos) or s.startswith("\\r", pos):
            pos += 2
            continue
        break
    return pos


def _normalize_human_value(obj: dict) -> None:
    conversations = obj.get("conversations")
    if not isinstance(conversations, list):
        return

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") != "human":
            continue

        value = turn.get("value")
        if isinstance(value, str) and "<image>\\n" in value:
            turn["value"] = value.replace("<image>\\n", "<image>\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fix a blobbed JSONL file where many JSON objects are concatenated "
            "together with a literal separator like \\\\n between objects."
        )
    )
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT_PATH, help="Path to blobbed input JSONL")
    ap.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH, help="Path to write fixed JSONL")
    args = ap.parse_args()
    input_path = args.input
    output_path = args.output

    # Read raw bytes to confirm the exact separator.
    with open(input_path, "rb") as f:
        preview = f.read(5_000_000)
    _print_first_separator_bytes(preview)

    # Parse concatenated JSON objects robustly.
    with open(input_path, "rb") as f:
        data = f.read()

    text = data.decode("utf-8")
    dec = json.JSONDecoder()

    pos = 0
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        while True:
            pos = _skip_separators(text, pos)
            if pos >= len(text):
                break

            # Expect a JSON object; if not, surface a useful error.
            if text[pos] != "{":
                context = text[pos : min(len(text), pos + 120)]
                raise ValueError(
                    f"Unexpected content at offset {pos}. Expected '{{'. "
                    f"Context: {context!r}"
                )

            obj, new_pos = dec.raw_decode(text, pos)
            _normalize_human_value(obj)
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            pos = new_pos

    print(f"Done! Wrote {count} objects to {output_path}")


if __name__ == "__main__":
    main()