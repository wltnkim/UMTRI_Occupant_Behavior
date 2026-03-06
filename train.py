"""
Train LSTM, MLP, or Transformer classifier on temporal feature sequences.

Input: .pt file with (X: [N, T, 48], y: [N]) from generate_sequences.py
Output: results/<timestamp>/ with best model, confusion matrix, F1 analysis, etc.
"""

import os
import argparse
import datetime
import time
import math
import json
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from tqdm import tqdm
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image
from thop import profile


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FlexibleFeatureDataset(Dataset):
    """Dataset that supports selecting subsets of the 48-dim feature vector.
    Feature layout: pose[0:34] + gaze[34:36] + fm[36:48]
    """
    def __init__(self, X, y, features):
        self.y = y
        self.X = self._select_features(X, features)

    def _select_features(self, X, features):
        selected = []
        if "pose" in features:
            selected.append(X[:, :, :34])
        if "gaze" in features:
            selected.append(X[:, :, 34:36])
        if "fm" in features:
            selected.append(X[:, :, 36:48])
        return torch.cat(selected, dim=-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        x = self.norm(output.mean(dim=1))
        return self.fc(x)


class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, nhead, num_encoder_layers, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = output.mean(dim=1)  # mean pooling
        return self.fc(output)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, bal_acc, file_path):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    with open(file_path, "w") as f:
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\nClassification Report:\n")
        f.write(report)


def plot_confusion_matrix_subset(y_true, y_pred, output_dir, top=True, n=20):
    class_counts = Counter(y_true)
    if top:
        subset = [cls for cls, _ in class_counts.most_common(n)]
        cmap, title_suffix = "Blues", f"Top-{n} Frequent"
    else:
        subset = [cls for cls, _ in sorted(class_counts.items(), key=lambda x: x[1])[:n]]
        cmap, title_suffix = "Oranges", f"Bottom-{n} Least Frequent"

    mask = [yt in subset for yt in y_true]
    yt_sub = [yt for yt, keep in zip(y_true, mask) if keep]
    yp_sub = [yp for yp, keep in zip(y_pred, mask) if keep]
    cm = confusion_matrix(yt_sub, yp_sub, labels=subset)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=subset, yticklabels=subset, cmap=cmap)
    plt.title(f"Confusion Matrix ({title_suffix} Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    tag = "top" if top else "bottom"
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{tag}{n}.png"))
    plt.close()


def save_f1_analysis(y_true, y_pred, output_dir):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_dict = {
        int(k): {"f1": v["f1-score"], "support": v["support"]}
        for k, v in report.items() if k.isdigit()
    }
    sorted_items = sorted(f1_dict.items(), key=lambda x: x[1]["f1"], reverse=True)
    result = {
        "top5": [{"class_id": k, "f1": round(v["f1"], 4), "support": v["support"]}
                 for k, v in sorted_items[:5]],
        "bottom5": [{"class_id": k, "f1": round(v["f1"], 4), "support": v["support"]}
                    for k, v in sorted_items[-5:]]
    }
    with open(os.path.join(output_dir, "f1_top5_bottom5.json"), "w") as f:
        json.dump(result, f, indent=2)


def save_error_images(y_true, y_pred, test_dataset, output_dir, max_cases=5):
    err_dir = os.path.join(output_dir, "error_images")
    os.makedirs(err_dir, exist_ok=True)
    error_indices = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    for i, idx in enumerate(error_indices[:max_cases]):
        x, _ = test_dataset[idx]
        x_img = x.permute(1, 0).unsqueeze(0)
        save_path = os.path.join(err_dir, f"error_{i+1}_true{y_true[idx]}_pred{y_pred[idx]}.png")
        save_image(x_img, save_path)


def measure_computational_cost(model, test_dataset, args, result_dir, device):
    x_sample, _ = test_dataset[0]
    x_sample = x_sample.unsqueeze(0).to(device)
    if args.model == "mlp":
        x_sample = x_sample.view(1, -1)

    flops, params = profile(model, inputs=(x_sample,), verbose=False)

    model.eval()
    with torch.no_grad():
        for _ in range(10):  # warm-up
            _ = model(x_sample.clone())
        timings = []
        for _ in range(100):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            _ = model(x_sample.clone())
            torch.cuda.synchronize() if device.type == 'cuda' else None
            timings.append((time.time() - start) * 1000)

    with open(os.path.join(result_dir, "computational_stats.txt"), "w") as f:
        f.write(f"FLOPs: {flops / 1e9:.4f} GFLOPs\n")
        f.write(f"Trainable Parameters: {params / 1e6:.4f} M\n")
        f.write(f"Inference Time: {np.mean(timings):.2f} +/- {np.std(timings):.2f} ms (100 runs)\n")


# ---------------------------------------------------------------------------
# Feature parsing
# ---------------------------------------------------------------------------
def parse_feature_selection(raw_string):
    allowed = {"pose", "gaze", "fm"}
    tokens = raw_string.lower().replace(" ", "").split("+")
    selected = set()
    for t in tokens:
        if t in allowed:
            selected.add(t)
        else:
            print(f"[WARNING] Unknown feature '{t}' ignored.")
    return selected


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    start_time = time.time()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_str = "+".join(sorted(args.features))
    result_dir = os.path.join("results", f"{now}_{feature_str}_{args.model}")
    os.makedirs(result_dir, exist_ok=True)

    # Save config
    with open(os.path.join(result_dir, "config.yaml"), 'w') as f:
        yaml.dump(vars(args), f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | Model: {args.model} | Features: {args.features}")

    # Load and split data
    X, y = torch.load(args.data_path)
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.2, random_state=42)
    val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)

    train_dataset = FlexibleFeatureDataset(train_X, train_y, args.features)
    val_dataset = FlexibleFeatureDataset(val_X, val_y, args.features)
    test_dataset = FlexibleFeatureDataset(test_X, test_y, args.features)
    print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    input_shape = train_dataset[0][0].shape
    is_sequential = args.model in ["lstm", "transformer"]
    input_dim = input_shape[1] if is_sequential else input_shape[0] * input_shape[1]
    num_classes = len(torch.unique(train_y))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Build model
    if args.model == "lstm":
        model = LSTMClassifier(input_dim, args.hidden_dim, num_classes,
                               num_layers=args.num_layers, dropout=args.dropout).to(device)
    elif args.model == "mlp":
        model = DeepMLP(input_dim, num_classes).to(device)
    elif args.model == "transformer":
        model = TransformerClassifier(input_dim, args.hidden_dim, num_classes,
                                      nhead=args.nhead,
                                      num_encoder_layers=args.num_encoder_layers,
                                      dropout=args.dropout).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {total_params / 1e6:.2f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []
        for x, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            x, y_batch = x.to(device), y_batch.to(device)
            if args.model == "mlp":
                x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_bal_acc = balanced_accuracy_score(all_labels, all_preds)

        # Validate
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                x, y_batch = x.to(device), y_batch.to(device)
                if args.model == "mlp":
                    x = x.view(x.size(0), -1)
                outputs = model(x)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train BalAcc={train_bal_acc:.4f} | "
              f"Val Loss={val_loss/len(val_loader):.4f}, "
              f"Val Acc={val_acc:.4f}, Val BalAcc={val_bal_acc:.4f}")

        if val_bal_acc > best_acc:
            best_acc = val_bal_acc
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pt"))
            patience_counter = 0
            print(f"  -> Saved best model (BalAcc={best_acc:.4f})")
        else:
            patience_counter += 1

        scheduler.step(val_loss)
        if patience_counter >= args.patience:
            print("[INFO] Early stopping triggered.")
            break

    # Test
    print("[INFO] Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(result_dir, "best_model.pt")))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y_batch in tqdm(test_loader, desc="[Test]"):
            x, y_batch = x.to(device), y_batch.to(device)
            if args.model == "mlp":
                x = x.view(x.size(0), -1)
            outputs = model(x)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"[RESULT] Test Accuracy: {acc:.4f}, Balanced Accuracy: {bal_acc:.4f}")

    # Save evaluation results
    save_confusion_matrix(all_labels, all_preds, bal_acc,
                          os.path.join(result_dir, "confusion_matrix.txt"))
    plot_confusion_matrix_subset(all_labels, all_preds, result_dir, top=True)
    plot_confusion_matrix_subset(all_labels, all_preds, result_dir, top=False)
    save_f1_analysis(all_labels, all_preds, result_dir)
    save_error_images(all_labels, all_preds, test_dataset, result_dir)
    measure_computational_cost(model, test_dataset, args, result_dir, device)

    elapsed = (time.time() - start_time) / 60
    print(f"[DONE] Total training time: {elapsed:.2f} minutes. Results saved to {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train behavior classifier (LSTM / MLP / Transformer)")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .pt file (e.g., features/pose_gaze_fm_30f10s_10.pt)")
    parser.add_argument("--features", type=str, default="pose+gaze+fm",
                        help="Feature combination: pose, gaze, fm (e.g., pose+fm)")

    # Model
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "mlp", "transformer"])
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for LSTM/Transformer")
    parser.add_argument("--num_layers", type=int, default=3, help="LSTM layers")
    parser.add_argument("--nhead", type=int, default=8, help="Transformer attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Transformer encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    args = parser.parse_args()
    args.features = parse_feature_selection(args.features)
    train(args)
