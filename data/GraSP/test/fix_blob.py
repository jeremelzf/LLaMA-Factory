import json

input_path = "/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/test/GraSP_test_blob.jsonl"
output_path = "/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/test/GraSP_test.jsonl"

with open(input_path, "r") as f:
    content = f.read()

# Use a streaming JSON decoder to extract objects one by one
decoder = json.JSONDecoder()
idx = 0
count = 0

with open(output_path, "w") as out:
    while idx < len(content):
        # Skip whitespace between objects
        while idx < len(content) and content[idx] in ' \t\r\n':
            idx += 1
        if idx >= len(content):
            break
        obj, end_idx = decoder.raw_decode(content, idx)
        out.write(json.dumps(obj) + "\n")
        idx = end_idx
        count += 1

print(f"Done! Wrote {count} objects.")