# mapping_generator.py
import os
import hashlib
import re
import json
from tqdm import tqdm

root_dir = "/home/superman/data/jkim/work/datasets/OBC_STRUCTURE_COPY"
mapping = {}

print("Generating hash-to-subjectID mapping...")
for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
    if "Color" in dirpath and any(fname.lower().endswith(".jpeg") for fname in filenames):
        # 경로에서 Subject ID (Hxxx) 추출
        match = re.search(r'(H\d{3})', dirpath)
        if match:
            subject_id = match.group(1)
            
            # .pkl 파일명에 사용된 것과 동일한 hash 생성
            dir_hash = hashlib.md5(dirpath.encode()).hexdigest()[:12]
            
            # 맵에 추가
            if dir_hash not in mapping:
                mapping[dir_hash] = subject_id

# 생성된 맵을 json 파일로 저장
output_path = "features/hash_to_subject_map.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"✅ Mapping saved to {output_path}")
# 결과 예시: {"abcdef123456": "H001", "fedcba654321": "H002", ...}