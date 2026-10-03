# UMTRI Occupant Behavior Recognition

[![Paper](https://img.shields.io/badge/Sensors-2025-blue)](https://www.mdpi.com/1424-8220/25/20/6323)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs25206323-green)](https://doi.org/10.3390/s25206323)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"Robust Occupant Behavior Recognition via Multimodal Sequence Modeling: A Comparative Study for In-Vehicle Monitoring Systems"**
Jisu Kim and Byoung-Keon D. Park, *Sensors* 2025, 25(20), 6323.
[[Paper]](https://www.mdpi.com/1424-8220/25/20/6323) · [[DOI]](https://doi.org/10.3390/s25206323)

A lightweight framework that recognizes in-vehicle occupant behavior from three hand-crafted visual features: **2D pose**, **2D gaze**, and **facial movement (action units)**. We compare MLP, LSTM, and Transformer temporal classifiers; the **Transformer encoder reaches the best Macro F1 of 0.9570** on the OBC dataset.

## Contents

- [Pipeline](#pipeline)
- [Results](#results)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Pipeline

```
Raw video frames
      |
      +--> YOLOv8-Pose   (34-d) --+
      +--> UniGaze       ( 2-d) --+--> merge + L2-normalize --> 48-d per frame
      +--> FMAE-IAT      (12-d) --+
                                        |
                                        v
                       sliding window + uniform subsampling
                                  [T, 48] sequences
                                        |
                       +----------------+----------------+
                       v                v                v
                      MLP             LSTM          Transformer
```

## Results

**Performance vs. efficiency** (Table 5 in the paper)

| Model | Config (L_span, S, L_samples) | Macro F1 | Params (M) | GFLOPs | Time (ms) |
|-------|-------------------------------|----------|------------|--------|-----------|
| MLP | Frame-level | 0.6705 | 0.06 | <0.001 | 0.08 |
| LSTM (Low-Cost) | (10, 10, 5) | 0.6601 | 1.39 | 0.01 | 0.17 |
| LSTM (High-Perf.) | (40, 10, 40) | 0.9931 | 1.39 | 0.05 | 0.44 |
| Transformer (Efficient) | (10, 5, 5) | 0.9395 | 4.24 | 0.02 | 0.33 |
| **Transformer (Best-Perf.)** | **(50, 10, 50)** | **0.9570** | **4.24** | **0.21** | **0.34** |

**Input modality ablation** (Table 3 in the paper; L_span=30, S=10, L_samples=30; Macro F1)

| Features | MLP | LSTM | Transformer |
|----------|-----|------|-------------|
| Pose | 0.6358 | 0.8784 | 0.8759 |
| Gaze | 0.0150 | 0.1145 | 0.1338 |
| FM | 0.2386 | 0.5425 | 0.7158 |
| Pose + Gaze | 0.6375 | 0.8875 | 0.8785 |
| Pose + FM | 0.6668 | 0.8838 | 0.9081 |
| Gaze + FM | 0.2663 | 0.5306 | 0.6611 |
| **Pose + Gaze + FM** | **0.6705** | **0.8941** | **0.8970** |

## Dataset

Experiments use the **Occupant Behavior Classification (OBC)** dataset, collected at the University of Michigan Transportation Research Institute (UMTRI).

- ~2.1M frames (10 fps, 1280×720), **79 behavior classes**, **42 participants** (21 M / 21 F)
- Recorded in a stationary 2018 Hyundai Genesis G90 with two Microsoft Azure Kinect sensors
- Three driving modes: Manual (MN), Semi-Automated (SA), Fully Automated (FA)
- Split: 80% train (1.68M frames), 10% validation (210K), 10% test (210K)

> **Note:** The OBC dataset is not publicly available for privacy reasons. The code can be adapted to any similarly structured dataset.

Expected layout:

```
<dataset_root>/
  <SubjectID>/
    OBC_<SubjectID>_<BehaviorCode>_<TrialNum>_.../
      Color/
        frame_0000.jpeg
        frame_0001.jpeg
        ...
```

## Installation

```bash
git clone https://github.com/wltnkim/UMTRI_Occupant_Behavior.git
cd UMTRI_Occupant_Behavior
pip install -r requirements.txt
```

Download the pre-trained models below and place them in the project root or `checkpoints/`.

| Model | Purpose | Dim | Source |
|-------|---------|-----|--------|
| `yolov8x-pose.pt` | 2D pose (17 keypoints) | 34 | [Ultralytics YOLOv8-Pose](https://docs.ultralytics.com/tasks/pose/) |
| `yolov8x-face-lindevs.pt` | Face detection | — | [YOLOv8-Face](https://github.com/akanametov/yolo-face) |
| UniGaze MAE-B16 | 2D gaze (pitch, yaw) | 2 | [UniGaze](https://github.com/lrdmurthy/unigaze) |
| `FMAE_ViT_base.pth` | Facial AU detection (12 AUs) | 12 | [FMAE-IAT](https://github.com/MSA-LMC/FMAE-IAT) |

## Usage

### 1. Extract per-frame features

```bash
# 2D pose: 17 keypoints (x, y) = 34-d
python feature_extraction/extract_pose.py \
    --data_root /path/to/dataset \
    --save_dir saved_frames_pose

# 2D gaze: pitch, yaw = 2-d
python feature_extraction/extract_gaze.py \
    --data_root /path/to/dataset \
    --save_dir saved_frames_gaze \
    --model_config /path/to/unigaze/configs/model/mae_b_16_gaze.yaml \
    --checkpoint /path/to/unigaze/checkpoints/unigaze_b16_joint.pth.tar \
    --face_model /path/to/unigaze/data/face_model.txt

# Facial movement: 12 action units
python feature_extraction/extract_facial_movement.py \
    --data_root /path/to/dataset \
    --save_dir saved_frames_au \
    --checkpoint checkpoints/FMAE_ViT_base.pth
```

### 2. Merge features

```bash
python data_processing/merge_features.py \
    --pose_dir saved_frames_pose \
    --gaze_dir saved_frames_gaze \
    --face_dir saved_frames_au \
    --output_path features/pose_gaze_fm_1frame.pt
```

### 3. Generate temporal sequences

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
| `--span_frames` | Temporal window size (e.g., 30 frames = 3 s at 10 fps) |
| `--sample_frames` | Frames uniformly subsampled from each span |
| `--step` | Sliding stride between windows |
| `--label_from` | `span` (majority vote, recommended), `sample`, or `center` |

### 4. Train a classifier

```bash
# LSTM
python train.py \
    --data_path features/pose_gaze_fm_30f10s_10.pt \
    --model lstm \
    --features pose+gaze+fm \
    --hidden_dim 256 --num_layers 3 \
    --batch_size 256 --epochs 200 --lr 1e-3 --patience 10

# Transformer (paper configuration)
python train.py \
    --data_path features/pose_gaze_fm_50f10s_50.pt \
    --model transformer \
    --features pose+gaze+fm \
    --hidden_dim 256 --nhead 8 --num_encoder_layers 4 \
    --dropout 0.2 \
    --batch_size 128 --epochs 200 --lr 1e-4 --patience 10

# MLP (single-frame baseline)
python train_mlp_singleframe.py \
    --data_path features/pose_gaze_fm_1frame.pt \
    --features pose+gaze+fm \
    --batch_size 256 --epochs 200 --lr 1e-3
```

`--features` accepts any combination of `pose`, `gaze`, `fm` joined by `+` (default: `pose+gaze+fm`), e.g. `pose+fm` or `pose`.

Each run saves its config, best model, confusion matrix, F1 analysis, and computational stats to `results/<timestamp>/`.

## Features

| Index | Feature | Dim | Extractor |
|-------|---------|-----|-----------|
| 0–33 | 2D body pose (17 keypoints x, y) | 34 | YOLOv8x-Pose |
| 34–35 | 2D gaze direction (pitch, yaw) | 2 | UniGaze (MAE-B16) |
| 36–47 | Facial action units (AU1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24) | 12 | FMAE-IAT (ViT-Base) |
| | **Concatenated per frame** | **48** | |

Each modality is L2-normalized per frame before concatenation.

## Project structure

```
├── feature_extraction/
│   ├── extract_pose.py              # YOLOv8-Pose keypoints
│   ├── extract_gaze.py              # UniGaze 2D gaze
│   └── extract_facial_movement.py   # FMAE-IAT action units
├── data_processing/
│   ├── merge_features.py            # fuse modalities into one .pt
│   ├── generate_sequences.py        # sliding window + uniform subsampling
│   └── mapping_generator.py         # hash-to-SubjectID mapping utility
├── train.py                         # LSTM / MLP / Transformer on sequences
├── train_mlp_singleframe.py         # MLP on single-frame features
└── requirements.txt
```

## Citation

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

## License

Released under the [MIT License](LICENSE).
