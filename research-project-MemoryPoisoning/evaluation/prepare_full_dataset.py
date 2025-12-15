import json
import os
import glob
import random

DATASET_DIR = "attack_dataset"
OUTPUT_FILE = "agent/attack_queries_full.json"
LIMIT_PER_FILE = 10
SEED = 42

def prepare_dataset():
    random.seed(SEED)
    print(f"Scanning {DATASET_DIR} for JSONL files...")
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jsonl")))
    
    all_attacks = {}
    
    # Add dummy entries to bypass main.py's skip logic (skips first 2 keys)
    all_attacks["dummy_skip_1"] = {"queries": [], "original_id": "", "target_id": ""}
    all_attacks["dummy_skip_2"] = {"queries": [], "original_id": "", "target_id": ""}
    
    attack_counter = 1
    
    for file_path in files:
        print(f"Processing {file_path}...")
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # Randomly sample if more than limit
        if len(lines) > LIMIT_PER_FILE:
            selected_lines = random.sample(lines, LIMIT_PER_FILE)
        else:
            selected_lines = lines
            
        for line in selected_lines:
                
            item = json.loads(line)
            attack_key = f"attack_{attack_counter}"
            
            queries = []
            raw_queries = item["attack_queries"]
            
            if isinstance(raw_queries, list):
                for j, q_text in enumerate(raw_queries):
                    queries.append({
                        "id": f"query_{j+1}",
                        "template": q_text,
                        "answer": ["Unknown"] # Placeholder as dataset lacks answers
                    })
            
            all_attacks[attack_key] = {
                "queries": queries,
                "original_id": item["original_id"],
                "target_id": item["target_id"]
            }
            attack_counter += 1
            
    print(f"Total attacks prepared: {len(all_attacks) - 2} (plus 2 dummies)")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_attacks, f, indent=4)
        
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_dataset()
