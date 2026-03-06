"""
Generate a hash-to-SubjectID mapping from the OBC dataset directory structure.
This is useful for subject-disjoint train/val/test splits.

Output: features/hash_to_subject_map.json
"""

import os
import argparse
import hashlib
import re
import json
from tqdm import tqdm


def main(args):
    mapping = {}

    print("Generating hash-to-SubjectID mapping...")
    for dirpath, dirnames, filenames in tqdm(os.walk(args.data_root)):
        if "Color" in dirpath and any(fname.lower().endswith(".jpeg") for fname in filenames):
            # Extract Subject ID (e.g., H001) from path
            match = re.search(r'(H\d{3})', dirpath)
            if match:
                subject_id = match.group(1)
                dir_hash = hashlib.md5(dirpath.encode()).hexdigest()[:12]
                if dir_hash not in mapping:
                    mapping[dir_hash] = subject_id

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[DONE] Mapping saved to {args.output_path} ({len(mapping)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hash-to-SubjectID mapping")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the OBC dataset")
    parser.add_argument("--output_path", type=str, default="features/hash_to_subject_map.json",
                        help="Output path for the mapping JSON file")
    args = parser.parse_args()
    main(args)
