"""
Merge per-frame features from three modalities (pose, gaze, facial movement)
into a single .pt file. Each modality is L2-normalized before concatenation.

Output: (X: [N, 48], y: [N]) where 48 = pose(34) + gaze(2) + face(12)
"""

import os
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm


def main(args):
    # Find common files across all three modalities
    pose_files = set(os.listdir(args.pose_dir))
    gaze_files = set(os.listdir(args.gaze_dir))
    face_files = set(os.listdir(args.face_dir))
    common_files = sorted(list(pose_files & gaze_files & face_files))
    print(f"[INFO] Found {len(common_files)} common .pkl files.")

    all_features = []
    all_labels = []

    for filename in tqdm(common_files, desc="Merging features"):
        try:
            with open(os.path.join(args.pose_dir, filename), "rb") as f:
                pose_feat, label_pose = pickle.load(f)
            with open(os.path.join(args.gaze_dir, filename), "rb") as f:
                gaze_feat, label_gaze = pickle.load(f)
            with open(os.path.join(args.face_dir, filename), "rb") as f:
                face_feat, label_face = pickle.load(f)

            # Label consistency check
            if not (label_pose == label_gaze == label_face):
                raise ValueError(f"Label mismatch in {filename}")

            if pose_feat.ndim != 1 or gaze_feat.ndim != 1 or face_feat.ndim != 1:
                raise ValueError(
                    f"Invalid shape in {filename}: "
                    f"pose={pose_feat.shape}, gaze={gaze_feat.shape}, face={face_feat.shape}")

            # L2-normalize each modality
            pose_feat = pose_feat / (np.linalg.norm(pose_feat) + 1e-8)
            gaze_feat = gaze_feat / (np.linalg.norm(gaze_feat) + 1e-8)
            face_feat = face_feat / (np.linalg.norm(face_feat) + 1e-8)

            # Concatenate: [pose:34 + gaze:2 + face:12] = 48 dims
            fused_feat = np.concatenate([pose_feat, gaze_feat, face_feat], axis=-1)
            all_features.append(fused_feat)
            all_labels.append(label_pose)

        except Exception as e:
            print(f"[WARNING] Skipping {filename}: {e}")
            continue

    X = torch.tensor(np.stack(all_features), dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save((X, y), args.output_path)

    print(f"[DONE] Saved fused dataset to {args.output_path}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multi-modal features into a single .pt file")
    parser.add_argument("--pose_dir", type=str, required=True,
                        help="Directory containing per-frame pose .pkl files")
    parser.add_argument("--gaze_dir", type=str, required=True,
                        help="Directory containing per-frame gaze .pkl files")
    parser.add_argument("--face_dir", type=str, required=True,
                        help="Directory containing per-frame AU .pkl files")
    parser.add_argument("--output_path", type=str, default="features/pose_gaze_fm_1frame.pt",
                        help="Output path for merged .pt file")
    args = parser.parse_args()
    main(args)
