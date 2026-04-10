import json

input_path = "/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/test/GraSP_test_blob.jsonl"
output_path = "/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/test/GraSP_test.jsonl"

with open(input_path, "r") as f:
    content = f.read()

# Split on literal \n between JSON objects and rewrite properly
objects = content.strip().split('\n')

with open(output_path, "w") as f:
    for obj in objects:
        obj = obj.strip()
        if obj:
            # Validate it's proper JSON then rewrite
            parsed = json.loads(obj)
            f.write(json.dumps(parsed) + "\n")

print("Done!")