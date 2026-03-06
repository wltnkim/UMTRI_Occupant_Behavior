"""
Extract facial movement features (12 Action Units) from video frames
using FMAE-IAT (ViT-Base) with YOLOv8-Face for face detection.
Outputs per-frame .pkl files containing (au_vector[12], label).

Requires:
    - FMAE-IAT checkpoint: https://github.com/MSA-LMC/FMAE-IAT
    - models_vit.py from the FMAE-IAT repository
    - YOLOv8-Face model: https://github.com/akanametov/yolo-face
"""

import os
import re
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle
import hashlib
from tqdm import tqdm
from ultralytics import YOLO
from models_vit import VisionTransformer

# Action Units extracted by FMAE-IAT
AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10",
            "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"]


def extract_behavior_class(folder_name):
    """Parse folder name to extract behavior class code."""
    match = re.match(r"OBC_H\d{3}_(.+?)_\d{3}_", folder_name)
    return match.group(1) if match else None


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load AU model (FMAE-IAT ViT-Base)
    model = VisionTransformer(
        img_size=224, patch_size=16, num_classes=12,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)

    # Load YOLOv8-Face for face detection
    face_detector = YOLO(args.face_model)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    os.makedirs(args.save_dir, exist_ok=True)

    # Resume support
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

    for dirpath, _, filenames in tqdm(os.walk(args.data_root)):
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

        frame_files = sorted([f for f in filenames if f.lower().endswith(".jpeg")])
        dir_hash = hashlib.md5(dirpath.encode()).hexdigest()[:12]

        for idx, frame_file in enumerate(frame_files):
            try:
                img_path = os.path.join(dirpath, frame_file)
                frame = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_detector(img_rgb, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                au_vector = np.zeros(12, dtype=np.float32)
                if len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
                    face_crop = img_rgb[y1:y2, x1:x2]

                    input_tensor = transform(face_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        scores = torch.sigmoid(output).squeeze().cpu().numpy()
                        au_vector = scores.astype(np.float32)

                save_path = os.path.join(args.save_dir, f"{dir_hash}_frame{idx}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump((au_vector, label), f)

            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

        with open(completed_log_path, "a") as f:
            f.write(dirpath + "\n")

    with open(label_map_path, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[DONE] AU features saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract facial AU features using FMAE-IAT")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the OBC dataset")
    parser.add_argument("--save_dir", type=str, default="saved_frames_au",
                        help="Directory to save per-frame .pkl files")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/FMAE_ViT_base.pth",
                        help="Path to FMAE-IAT ViT-Base checkpoint")
    parser.add_argument("--face_model", type=str, default="yolov8x-face-lindevs.pt",
                        help="Path to YOLOv8-Face model")
    args = parser.parse_args()
    main(args)
