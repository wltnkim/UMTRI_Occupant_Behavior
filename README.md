# Robust Occupant Behavior Recognition via Multimodal Sequence Modeling

Official implementation of:

> **Robust Occupant Behavior Recognition via Multimodal Sequence Modeling: A Comparative Study for In-Vehicle Monitoring Systems**
>
> Jisu Kim and Byoung-Keon D. Park
>
> *Sensors*, 2025, 25(20), 6323 &nbsp; | &nbsp; [Paper](https://www.mdpi.com/1424-8220/25/20/6323) &nbsp; | &nbsp; DOI: [10.3390/s25206323](https://doi.org/10.3390/s25206323)

This repository provides code for a lightweight framework that recognizes in-vehicle occupant behaviors using three hand-crafted visual features: **2D Pose**, **2D Gaze**, and **Facial Movement (Action Units)**. We compare three temporal classifiers — MLP, LSTM, and Transformer — and show that the **Transformer encoder achieves the best Macro F1 of 0.9570** on the OBC dataset.

---

## Pipeline Overview

```
Raw Video Frames
       |
       v
+------+------+------+
|      |      |      |
v      v      v      v
YOLO   Uni    FMAE   (per-frame feature extraction)
Pose   Gaze   -IAT
(34d)  (2d)   (12d)
|      |      |
+------+------+
       |
       v
  Merge & L2-Normalize
  -> 48-dim feature vector per frame
       |
       v
  Sliding Window + Uniform Subsampling
  -> Temporal sequences [T, 48]
       |
       v
  +----+----+-------------+
  |         |             |
  v         v             v
 MLP      LSTM      Transformer
(static) (recurrent) (attention)
```

---

## Results

Performance on the OBC dataset (79 classes, random 80/10/10 split):

| Model | Features | Seq. Config | Macro F1 |
|-------|----------|-------------|----------|
| MLP | Pose+Gaze+FM | 1 frame (static) | 0.8474 |
| LSTM | Pose+Gaze+FM | 30 frames | 0.9931 |
| **Transformer** | **Pose+Gaze+FM** | **span=50, sample=25** | **0.9570** |

### Feature Ablation (LSTM, 30-frame sequences)

| Features | Dim | Macro F1 |
|----------|-----|----------|
| Pose only | 34 | 0.8780 |
| Pose + Gaze | 36 | 0.9069 |
| Pose + FM | 46 | 0.9069 |
| Pose + Gaze + FM | 48 | 0.8996 |

---

## Dataset

All experiments use the **Occupant Behavior Classification (OBC)** dataset, collected at the University of Michigan Transportation Research Institute (UMTRI).

- ~2.1 million frames across **79 behavior classes** from **42 participants**
- Recorded in a controlled, stationary vehicle environment
- **Note**: The OBC dataset is not publicly available. This code can be adapted to any similarly structured dataset.

### Expected Data Structure

```
<dataset_root>/
  <SubjectID>/
    OBC_<SubjectID>_<BehaviorCode>_<TrialNum>_.../
      Color/
        frame_0000.jpeg
        frame_0001.jpeg
        ...
```

---

## Installation

```bash
git clone https://github.com/wltnkim/UMTRI_Occupant_Behavior.git
cd UMTRI_Occupant_Behavior
pip install -r requirements.txt
```

### Pre-trained Models Required

Download the following and place them in the project root or `checkpoints/` directory:

| Model | Purpose | Dim | Source |
|-------|---------|-----|--------|
| `yolov8x-pose.pt` | 2D Pose (17 keypoints) | 34 | [Ultralytics YOLOv8-Pose](https://docs.ultralytics.com/tasks/pose/) |
| `yolov8x-face-lindevs.pt` | Face Detection | — | [YOLOv8-Face](https://github.com/akanametov/yolo-face) |
| UniGaze MAE-B16 | 2D Gaze (pitch, yaw) | 2 | [UniGaze](https://github.com/lrdmurthy/unigaze) |
| `FMAE_ViT_base.pth` | Facial AU Detection (12 AUs) | 12 | [FMAE-IAT](https://github.com/MSA-LMC/FMAE-IAT) |

---

## Usage

### Step 1: Feature Extraction

Extract per-frame features from each modality independently.

**2D Pose (YOLOv8-Pose)** — 17 keypoints (x, y) = 34 dimensions:
```bash
python feature_extraction/extract_pose.py \
    --data_root /path/to/dataset \
    --save_dir saved_frames_pose
```

**2D Gaze (UniGaze)** — pitch and yaw = 2 dimensions:
```bash
python feature_extraction/extract_gaze.py \
    --data_root /path/to/dataset \
    --save_dir saved_frames_gaze \
    --model_config /path/to/unigaze/configs/model/mae_b_16_gaze.yaml \
    --checkpoint /path/to/unigaze/checkpoints/unigaze_b16_joint.pth.tar \
    --face_model /path/to/unigaze/data/face_model.txt
```

**Facial Movement / Action Units (FMAE-IAT)** — 12 AUs:
```bash
python feature_extraction/extract_facial_movement.py \
    --data_root /path/to/dataset \
    --save_dir saved_frames_au \
    --checkpoint checkpoints/FMAE_ViT_base.pth
```

### Step 2: Merge Features

Fuse per-frame features from all three modalities into a single `.pt` file:

```bash
python data_processing/merge_features.py \
    --pose_dir saved_frames_pose \
    --gaze_dir saved_frames_gaze \
    --face_dir saved_frames_au \
    --output_path features/pose_gaze_fm_1frame.pt
```

### Step 3: Generate Temporal Sequences

Convert single-frame features into temporal sequences via sliding window with uniform subsampling:

```bash
python data_processing/generate_sequences.py \
    --data_path features/pose_gaze_fm_1frame.pt \
    --span_frames 30 \
    --sample_frames 10 \
    --step 10 \
    --save_path features/pose_gaze_fm_30f10s_10.pt \
    --label_from span
```

| Argument | Description |
|----------|-------------|
| `--span_frames` | Temporal window size (e.g., 30 frames = 3s at 10 fps) |
| `--sample_frames` | Frames to uniformly subsample from each span |
| `--step` | Sliding stride between windows |
| `--label_from` | `span` (majority vote, recommended), `sample`, or `center` |

### Step 4: Train Classifier

**LSTM:**
```bash
python train.py \
    --data_path features/pose_gaze_fm_30f10s_10.pt \
    --model lstm \
    --features pose+gaze+fm \
    --hidden_dim 256 --num_layers 3 \
    --batch_size 256 --epochs 200 --lr 1e-3 --patience 10
```

**Transformer:**
```bash
python train.py \
    --data_path features/pose_gaze_fm_30f10s_10.pt \
    --model transformer \
    --features pose+gaze+fm \
    --hidden_dim 256 --nhead 8 --num_encoder_layers 3 \
    --batch_size 256 --epochs 200 --lr 1e-3 --patience 10
```

**MLP (single-frame baseline):**
```bash
python train_mlp_singleframe.py \
    --data_path features/pose_gaze_fm_1frame.pt \
    --features pose+gaze+fm \
    --batch_size 256 --epochs 200 --lr 1e-3
```

**Feature selection** — any combination of `pose`, `gaze`, `fm`:
```bash
--features pose+fm       # Pose + Facial Movement only
--features pose          # Pose only
--features pose+gaze+fm  # All features (default)
```

Results (config, best model, confusion matrix, F1 analysis, computational stats) are saved to `results/<timestamp>/`.

---

## Feature Description

| Index | Feature | Dim | Extractor |
|-------|---------|-----|-----------|
| 0-33 | 2D Body Pose (17 keypoints x, y) | 34 | YOLOv8x-Pose |
| 34-35 | 2D Gaze Direction (pitch, yaw) | 2 | UniGaze (MAE-B16) |
| 36-47 | Facial Action Units (AU1,2,4,6,7,10,12,14,15,17,23,24) | 12 | FMAE-IAT (ViT-Base) |
| **Total** | **Concatenated per frame** | **48** | |

All features are L2-normalized per frame before concatenation.

---

## Project Structure

```
UMTRI_Occupant_Behavior/
├── feature_extraction/
│   ├── extract_pose.py              # YOLOv8-Pose keypoint extraction
│   ├── extract_gaze.py              # UniGaze 2D gaze estimation
│   └── extract_facial_movement.py   # FMAE-IAT Action Unit detection
├── data_processing/
│   ├── merge_features.py            # Fuse multi-modal features into .pt
│   ├── generate_sequences.py        # Sliding window + uniform subsampling
│   └── mapping_generator.py         # Hash-to-SubjectID mapping utility
├── train.py                         # Train LSTM / MLP / Transformer (sequences)
├── train_mlp_singleframe.py         # Train MLP on single-frame features
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{kim2025robust,
  title     = {Robust Occupant Behavior Recognition via Multimodal Sequence Modeling:
               A Comparative Study for In-Vehicle Monitoring Systems},
  author    = {Kim, Jisu and Park, Byoung-Keon D.},
  journal   = {Sensors},
  volume    = {25},
  number    = {20},
  pages     = {6323},
  year      = {2025},
  doi       = {10.3390/s25206323}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
