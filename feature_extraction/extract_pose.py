"""
Extract 2D pose keypoints from video frames using YOLOv8-Pose.
Outputs per-frame .pkl files containing (pose_vector[34], label).
"""

import os
import re
import argparse
import numpy as np
import pickle
import hashlib
from tqdm import tqdm
from ultralytics import YOLO


def extract_pose_from_image(model, img_path):
    """Extract 2D pose keypoints from a single image.
    Returns a flattened array of (x, y) for 17 keypoints (34 dims).
    """
    results = model(img_path, verbose=False)
    if len(results[0].keypoints.xy) == 0:
        return np.zeros((17, 2), dtype=np.float32).flatten()
    return results[0].keypoints.xy[0].cpu().numpy().flatten()


def extract_behavior_class(folder_name):
    """Parse folder name to extract behavior class code.
    Example: 'OBC_H001_WALKING_001_' -> 'WALKING'
    """
    match = re.match(r"OBC_H\d{3}_(.+?)_\d{3}_", folder_name)
    return match.group(1) if match else None


def main(args):
    model = YOLO("yolov8x-pose.pt")

    os.makedirs(args.save_dir, exist_ok=True)

    # Resume support: track completed directories
    completed_log_path = os.path.join(args.save_dir, "completed_paths.txt")
    if os.path.exists(completed_log_path):
        with open(completed_log_path, "r") as f:
            completed_paths = set(line.strip() for line in f.readlines())
    else:
        completed_paths = set()

    label_map = {}
    label_counter = 0
    label_map_path = os.path.join(args.save_dir, "label_map.pkl")
    if os.path.exists(label_map_path):
        with open(label_map_path, "rb") as f:
            label_map = pickle.load(f)
        label_counter = len(label_map)

    for dirpath, dirnames, filenames in tqdm(os.walk(args.data_root)):
        if "Color" not in dirpath:
            continue
        if not any(fname.lower().endswith(".jpeg") for fname in filenames):
            continue
        if dirpath in completed_paths:
            continue

        parts = dirpath.split(os.sep)
        folder_name = next((p for p in parts if p.startswith("OBC_H")), None)
        if not folder_name:
            continue

        class_code = extract_behavior_class(folder_name)
        if not class_code:
            continue

        if class_code not in label_map:
            label_map[class_code] = label_counter
            label_counter += 1
        label = label_map[class_code]

        try:
            frame_files = sorted([f for f in filenames if f.lower().endswith(".jpeg")])
            dir_hash = hashlib.md5(dirpath.encode()).hexdigest()[:12]

            for idx, frame_file in enumerate(frame_files):
                img_path = os.path.join(dirpath, frame_file)
                pose = extract_pose_from_image(model, img_path)
                save_path = os.path.join(args.save_dir, f"{dir_hash}_frame{idx}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump((pose, label), f)

            with open(completed_log_path, "a") as f:
                f.write(dirpath + "\n")

        except Exception as e:
            print(f"[ERROR] {dirpath}: {e}")

    with open(label_map_path, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[DONE] Pose features saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2D pose features using YOLOv8-Pose")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the OBC dataset")
    parser.add_argument("--save_dir", type=str, default="saved_frames_pose",
                        help="Directory to save per-frame .pkl files")
    args = parser.parse_args()
    main(args)
