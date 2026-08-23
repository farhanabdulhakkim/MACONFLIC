"""
prepare_dataset.py - Generates recording-level train/val/test splits safely using torchaudio.
"""
import os
import glob
import random
import csv
import torchaudio

DATA_DIR = "dataset"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
SEED = 42

def get_source_id(filename, category):
    base_name = os.path.splitext(filename)[0]
    # Handle duplicate files identified in audit
    if "copy" in base_name.lower():
        return "EXCLUDE"
    if category == "non_elephant":
        # ESC-50 format: {fold}-{clipID}-{take}-{class}.wav -> group by {fold}-{clipID}
        parts = base_name.split('-')
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1]}"
    return base_name

def main():
    random.seed(SEED)
    manifest_data = []
    source_groups = {"elephant": {}, "non_elephant": {}}
    excluded_count = 0
    
    # 1. Scan and Group by Source ID
    for label_val, category in enumerate(["non_elephant", "elephant"]):
        cat_dir = os.path.join(DATA_DIR, category)
        for filepath in glob.glob(os.path.join(cat_dir, "**/*.*"), recursive=True):
            if not filepath.endswith((".wav", ".mp3", ".flac", ".ogg")): 
                continue
            
            filename = os.path.basename(filepath)
            source_id = get_source_id(filename, category)
            
            if source_id == "EXCLUDE":
                excluded_count += 1
                print(f"[EXCLUDE] Duplicate file skipped: {filename}")
                continue
                
            # Efficient header metadata inspection via torchaudio
            info = torchaudio.info(filepath)
            duration = round(info.num_frames / info.sample_rate, 4)
            sample_rate = info.sample_rate
            
            if source_id not in source_groups[category]:
                source_groups[category][source_id] = []
                
            source_groups[category][source_id].append({
                "filepath": filepath.replace('\\', '/'),
                "label": label_val,
                "category": category,
                "source_id": source_id,
                "duration": duration,
                "sr": sample_rate
            })

    # 2. Recording-Level Split (70% Train, 15% Val, 15% Test)
    splits = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_counts = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}
    
    for category, groups in source_groups.items():
        group_keys = list(groups.keys())
        random.shuffle(group_keys)
        
        n_total = len(group_keys)
        n_train = int(n_total * splits["train"])
        n_val = int(n_total * splits["val"])
        
        for i, source_id in enumerate(group_keys):
            if i < n_train:
                split_name = "train"
            elif i < n_train + n_val:
                split_name = "val"
            else:
                split_name = "test"
                
            for item in groups[source_id]:
                manifest_data.append((
                    item["filepath"], item["label"], item["category"], 
                    item["source_id"], item["duration"], item["sr"], split_name
                ))
                split_counts[split_name][item["label"]] += 1

    # 3. Save to manifest.csv
    with open(MANIFEST_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "category", "source_id", "duration", "sample_rate", "split"])
        writer.writerows(manifest_data)
        
    print(f"\nManifest successfully created: {MANIFEST_PATH}")
    print(f"Total files processed: {len(manifest_data)} (Excluded duplicates: {excluded_count})")
    print("\n--- Split & Class Distribution ---")
    for s, counts in split_counts.items():
        print(f"[{s.upper()}] Non-Elephant (0): {counts[0]} | Elephant (1): {counts[1]} | Total: {counts[0] + counts[1]}")

if __name__ == "__main__":
    main()