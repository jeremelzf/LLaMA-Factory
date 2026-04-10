import json

input_path = "/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/test/GraSP_test_blob.jsonl"
output_path = "/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/test/GraSP_test.jsonl"

with open(input_path, "r") as f:
    content = f.read()

# The separator is \\n (two chars: backslash + n)
parts = content.split('\\n')

count = 0
with open(output_path, "w") as out:
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            obj = json.loads(part)
            out.write(json.dumps(obj) + "\n")
            count += 1
        except json.JSONDecodeError as e:
            print(f"Skipping bad part at object {count}: {e}")

print(f"Done! Wrote {count} objects.")