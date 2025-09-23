# Robust Occupant Behavior Recognition via Multimodal Sequence Modeling

This repository contains the official source code and implementation for the paper: "Robust Occupant Behavior Recognition via Multimodal Sequence Modeling: A Comparative Study for In-Vehicle Monitoring Systems".

---

## 📜 Overview

This study investigates a lightweight and robust framework for in-vehicle occupant behavior recognition using multimodal visual features. We fuse sequential inputs extracted from **2D pose**, **2D gaze**, and **facial movements** to classify 79 distinct occupant behaviors. We provide a comparative analysis between a static Multi-Layer Perceptron (MLP) model and a temporal Long Short-Term Memory (LSTM) network, demonstrating the superiority of temporal modeling for this task.

Our LSTM-based approach achieves a macro F1-score of up to **0.9931** on the Occupant Behavior Classification (OBC) dataset.

---

## 🔧 Methodology

Our recognition pipeline consists of three main stages:

1.  **Feature Extraction**: For each video frame, we extract three feature types using powerful, pre-trained models:
    * **2D Pose**: 17 keypoints from YOLOv8-Pose.
    * **2D Gaze**: Pitch and Yaw from UniGaze.
    * **Facial Movement (FM)**: A 12-dimensional vector of Action Units (AUs) from FMAE-IAT.
2.  **Sequence Construction**: The extracted features are concatenated per frame, and these frames are grouped into fixed-length, overlapping sequences.
3.  **Temporal Classification**: The resulting sequences are fed into a lightweight classifier (MLP or LSTM) for behavior recognition.

---

## 💾 Dataset

All experiments are conducted on the **Occupant Behavior Classification (OBC) dataset**.

-   This dataset was originally collected at the University of Michigan Transportation Research Institute (UMTRI) as part of a sponsored research project (IRB: HUM00162942).
-   It contains approximately 2.1 million frames across 79 behavior classes, collected from 42 participants in a controlled, stationary vehicle environment.
-   **Note**: This dataset is not publicly available. This repository provides the code to train and evaluate models if you have access to the OBC dataset or a similarly structured dataset.

---

## ⚙️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/wltnkim/UMTRI_Occupant_Behavior.git
    cd UMTRI_Occupant_Behavior
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    conda env create -f environment.yml
    ```

3.  Using Pip: Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

The pipeline is executed in the following order:

1.  **Feature Extraction**

    Run the feature extraction script on the raw video data.
    ```bash
    python extract_features.py --video_root [path_to_obc_videos] --output_dir [path_to_save_features]
    ```

2.  **Training**

    Train the classifier using the extracted features. Key hyperparameters can be adjusted via command-line arguments.
    ```bash
    python train.py --feature_dir [path_to_saved_features] --model_type LSTM --seq_length 40 --step_size 10 --save_path ./models/
    ```
    -   `--model_type`: Choose between `MLP` and `LSTM`.
    -   `--seq_length`: The number of frames per sequence.
    -   `--step_size`: The step size (stride) for creating sequences.
    -   Other hyperparameters like learning rate, batch size, etc., can be found in `train.py`.

3.  **Evaluation**

    Evaluate the trained model on the test set.
    ```bash
    python evaluate.py --model_path ./models/[trained_model_name].pth --feature_dir [path_to_saved_features] --test_split
    ```

## 📧 Contact
For any questions, please contact Jisu Kim at jkim73@huskers.unl.edu. 

---

<!-- ## 📄 Citation

If you find our work useful, please consider citing our paper:
```bibtex
@article{kim2025occupant,
  title   = {Robust Occupant Behavior Recognition via Multimodal Sequence Modeling: A Comparative Study for In-Vehicle Monitoring Systems},
  author  = {Kim, Jisu and Park, Byoung-Keon D.},
  journal = {Sensors},
  year    = {2025},
  note    = {Manuscript submitted for publication}
} -->
