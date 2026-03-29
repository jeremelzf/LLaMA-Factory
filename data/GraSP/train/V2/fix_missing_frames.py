import json
import os
import glob

def find_nearest_frame(missing_path: str) -> str:
    """Find the nearest existing frame to replace a missing one."""
    directory = os.path.dirname(missing_path)
    filename = os.path.basename(missing_path)
    frame_num = int(filename.replace('.jpg', ''))
    
    # Search within ±300 frames (10 seconds at 30fps)
    for offset in range(1, 300):
        for delta in [offset, -offset]:
            candidate_num = frame_num + delta
            if candidate_num < 0:
                continue
            candidate = os.path.join(directory, f"{candidate_num:09d}.jpg")
            if os.path.exists(candidate):
                return candidate
    
    return None  # No neighbour found within range

input_path = '/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/train/V2/GraSP_PhaseStep_Recognition_updated.jsonl'
output_path = '/nfs/home/svu/e0957602/LLaMA-Factory/data/GraSP/train/V2/GraSP_PhaseStep_Recognition_final.jsonl'

total = 0
replaced = 0
skipped = 0
no_neighbour = 0

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        entry = json.loads(line)
        img_path = entry['image'][0]
        total += 1
        
        if not os.path.exists(img_path):
            nearest = find_nearest_frame(img_path)
            if nearest:
                entry['image'][0] = nearest
                replaced += 1
            else:
                no_neighbour += 1
                skipped += 1
                continue  # Skip only if truly no neighbour exists
        
        fout.write(json.dumps(entry) + '\n')

print(f"Total entries: {total}")
print(f"Replaced with neighbour: {replaced}")
print(f"Skipped (no neighbour found): {skipped}")
print(f"Output entries: {total - skipped}")