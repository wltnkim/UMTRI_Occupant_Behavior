"""
Extract 2D gaze features (pitch, yaw) from video frames using UniGaze.
Outputs per-frame .pkl files containing (gaze_vector[2], label).

Requires the UniGaze repository to be installed:
    https://github.com/lrdmurthy/unigaze
"""

import os
import re
import argparse
import cv2
import numpy as np
import pickle
import hashlib
from tqdm import tqdm
import torch
import face_alignment
from torchvision import transforms
from omegaconf import OmegaConf
from gazelib.gaze.normalize import estimateHeadPose, normalize
from gazelib.gaze.gaze_utils import pitchyaw_to_vector, vector_to_pitchyaw
from gazelib.label_transform import get_face_center_by_nose
from utils import instantiate_from_cfg


def set_dummy_camera_model(image):
    """Create a simple pinhole camera model from image dimensions."""
    h, w = image.shape[:2]
    focal_length = w * 4
    center = (w // 2, h // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")
    camera_distortion = np.zeros((1, 5))
    return camera_matrix, camera_distortion


def extract_gaze_from_image(image, model, fa, face_model_path, device, image_transform):
    """Extract 2D gaze direction (pitch, yaw) from a single image."""
    preds = fa.get_landmarks(image)
    if preds is None:
        return np.zeros(2, dtype=np.float32)

    landmarks = preds[0].astype(float)
    camera_matrix, camera_distortion = set_dummy_camera_model(image)
    face_model_load = np.loadtxt(face_model_path)
    face_model = face_model_load[[20, 23, 26, 29, 15, 19], :]
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :].reshape(6, 1, 2)
    hr, ht = estimateHeadPose(landmarks_sub, face_model.reshape(6, 1, 3),
                               camera_matrix, camera_distortion)
    face_center_camera_cord, _ = get_face_center_by_nose(
        hR=cv2.Rodrigues(hr)[0], ht=ht, face_model_load=face_model_load)

    img_normalized, R, _, _, _, _ = normalize(
        image, landmarks, 960, 600, (224, 224),
        face_center_camera_cord, hr, ht, camera_matrix)

    input_var = img_normalized[:, :, [2, 1, 0]]  # BGR -> RGB
    input_var = image_transform(input_var).unsqueeze(0).float().to(device)
    ret = model(input_var)

    pred_gaze = ret["pred_gaze"][0].cpu().data.numpy()
    return pred_gaze.flatten()


def extract_behavior_class(folder_name):
    """Parse folder name to extract behavior class code."""
    match = re.match(r"OBC_H\d{3}_(.+?)_\d{3}_", folder_name)
    return match.group(1) if match else None


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load UniGaze model
    pretrained_model_cfg = OmegaConf.load(args.model_config)['net_config']
    pretrained_model_cfg.params.custom_pretrained_path = None
    model = instantiate_from_cfg(pretrained_model_cfg)
    model.load_state_dict(torch.load(args.checkpoint)['model_state'])
    model.eval()
    model.to(device)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

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
                image = cv2.imread(img_path)
                gaze = extract_gaze_from_image(
                    image, model, fa, args.face_model, device, image_transform)
                save_path = os.path.join(args.save_dir, f"{dir_hash}_frame{idx}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump((gaze, label), f)

            with open(completed_log_path, "a") as f:
                f.write(dirpath + "\n")

        except Exception as e:
            print(f"[ERROR] {dirpath}: {e}")

    with open(label_map_path, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[DONE] Gaze features saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2D gaze features using UniGaze")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the OBC dataset")
    parser.add_argument("--save_dir", type=str, default="saved_frames_gaze",
                        help="Directory to save per-frame .pkl files")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to UniGaze model config YAML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to UniGaze checkpoint (.pth.tar)")
    parser.add_argument("--face_model", type=str, required=True,
                        help="Path to face_model.txt for head pose estimation")
    args = parser.parse_args()
    main(args)
