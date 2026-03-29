import json
import os

input_path = '/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/train/V2/GraSP_PhaseStep_Recognition_updated.jsonl'
output_path = '/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/train/V2/GraSP_PhaseStep_Recognition_final.jsonl'

total = 0
already_exists = 0
replaced_1 = 0      # replaced with ±1 neighbour
replaced_2 = 0      # replaced with ±2 neighbour
skipped_1fps = 0    # 1fps entries that were missing — skipped entirely
skipped_no_neighbour = 0  # 30fps missing but no neighbour within ±2

def find_nearest_frame(missing_path: str, max_offset: int = 2):
    """Find nearest existing frame within max_offset. Returns (path, offset) or (None, None)."""
    directory = os.path.dirname(missing_path)
    filename = os.path.basename(missing_path)
    frame_num = int(filename.replace('.jpg', ''))

    for offset in range(1, max_offset + 1):
        for delta in [offset, -offset]:
            candidate_num = frame_num + delta
            if candidate_num < 0:
                continue
            candidate = os.path.join(directory, f"{candidate_num:09d}.jpg")
            if os.path.exists(candidate):
                return candidate, offset
    return None, None

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        entry = json.loads(line)
        img_path = entry['image'][0]
        total += 1

        if os.path.exists(img_path):
            already_exists += 1
            fout.write(json.dumps(entry) + '\n')
            continue

        # Determine if this is a 1fps or 30fps path
        is_1fps = '1fps_frames' in img_path

        if is_1fps:
            # Skip missing 1fps frames entirely — neighbour too visually different
            skipped_1fps += 1
            continue

        # 30fps: find nearest neighbour within ±2
        nearest, offset = find_nearest_frame(img_path, max_offset=2)

        if nearest is None:
            skipped_no_neighbour += 1
            continue

        entry['image'][0] = nearest
        if offset == 1:
            replaced_1 += 1
        elif offset == 2:
            replaced_2 += 1

        fout.write(json.dumps(entry) + '\n')

output_total = already_exists + replaced_1 + replaced_2
print(f"Total entries processed:        {total}")
print(f"Already existing (no change):   {already_exists}")
print(f"Replaced with ±1 neighbour:     {replaced_1}")
print(f"Replaced with ±2 neighbour:     {replaced_2}")
print(f"Skipped (missing 1fps frame):   {skipped_1fps}")
print(f"Skipped (no ±2 neighbour found):{skipped_no_neighbour}")
print(f"─────────────────────────────────────────")
print(f"Output entries written:         {output_total}")
print(f"Total skipped:                  {skipped_1fps + skipped_no_neighbour}")