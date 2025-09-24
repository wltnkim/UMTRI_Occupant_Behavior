import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

pose_dir = "/home/superman/data/jkim/work/side_project/UMTRI/2D_pose_classifier/YOLO-Pose/saved_frames_yolopose_align_1frame"
gaze_dir = "/home/superman/data/jkim/work/side_project/UMTRI/UniGaze/unigaze/saved_frames_unigaze_align_1frame"
face_dir = "/home/superman/data/jkim/work/side_project/UMTRI/FMAE-IAT/saved_frames_au_features"


common_files = sorted(list(set(os.listdir(pose_dir)) & set(os.listdir(gaze_dir)) & set(os.listdir(face_dir))))
print(f"[INFO] Found {len(common_files)} common .pkl files.")

all_features = []
all_labels = []

for filename in tqdm(common_files, desc="Processing"):
    try:
        # 🔸 Load each modality
        with open(os.path.join(pose_dir, filename), "rb") as f:
            pose_feat, label_pose = pickle.load(f)

        with open(os.path.join(gaze_dir, filename), "rb") as f:
            gaze_feat, label_gaze = pickle.load(f)

        with open(os.path.join(face_dir, filename), "rb") as f:
            face_feat, label_face = pickle.load(f)

        # 🔸 Label consistency check
        if not (label_pose == label_gaze == label_face):
            raise ValueError(f"Label mismatch in {filename}")

        # 🔸 Shape check
        if pose_feat.ndim != 1 or gaze_feat.ndim != 1 or face_feat.ndim != 1:
            raise ValueError(f"Invalid feature shape in {filename}: "
                             f"pose={pose_feat.shape}, gaze={gaze_feat.shape}, face={face_feat.shape}")

        # 🔸 Normalize (L2) — frame-wise
        pose_feat = pose_feat / (np.linalg.norm(pose_feat) + 1e-8)
        gaze_feat = gaze_feat / (np.linalg.norm(gaze_feat) + 1e-8)
        face_feat = face_feat / (np.linalg.norm(face_feat) + 1e-8)

        # 🔸 Concatenate
        fused_feat = np.concatenate([pose_feat, gaze_feat, face_feat], axis=-1)  # → shape (48,)
        all_features.append(fused_feat)
        all_labels.append(label_pose)

    except Exception as e:
        print(f"[WARNING] Skipping {filename}: {e}")
        continue

# 🔹 Stack and save
X = torch.tensor(np.stack(all_features), dtype=torch.float32)  # (N, 48)
y = torch.tensor(all_labels, dtype=torch.long)                # (N,)

output_path = "features/pose_gaze_face_dataset_1frame.pt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save((X, y), output_path)

print(f"[INFO] ✅ Saved fused dataset to {output_path}")
print(f"[INFO] 🔍 X shape: {X.shape}, y shape: {y.shape}")
