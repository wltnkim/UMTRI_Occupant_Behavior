"""
Train MLP or LSTM on single-frame features (no temporal dimension).

Input: .pt file with (X: [N, 48], y: [N]) from merge_features.py
Output: results/<timestamp>/ with best model and confusion matrix.
"""

import os
import argparse
import datetime
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FlexibleFeatureDataset(Dataset):
    """Dataset for single-frame features. Shape: [N, D] where D <= 48."""
    def __init__(self, X, y, features):
        self.y = y
        self.X = self._select_features(X, features)

    def _select_features(self, X, features):
        selected = []
        if "pose" in features:
            selected.append(X[:, :34])
        if "gaze" in features:
            selected.append(X[:, 34:36])
        if "fm" in features:
            selected.append(X[:, 36:48])
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
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out)


class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, file_path):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    with open(file_path, "w") as f:
        f.write("Confusion Matrix:\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\nClassification Report:\n")
        f.write(report)


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
# Training
# ---------------------------------------------------------------------------
def train(args):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_str = "+".join(sorted(args.features))
    result_dir = os.path.join("results", f"{now}_{feature_str}_{args.model}_1frame")
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "config.yaml"), 'w') as f:
        yaml.dump(vars(args), f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | Model: {args.model} | Features: {args.features}")

    X, y = torch.load(args.data_path)
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.2, random_state=42)
    val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)

    train_dataset = FlexibleFeatureDataset(train_X, train_y, args.features)
    val_dataset = FlexibleFeatureDataset(val_X, val_y, args.features)
    test_dataset = FlexibleFeatureDataset(test_X, test_y, args.features)
    print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(torch.unique(train_y))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    if args.model == "lstm":
        model = LSTMClassifier(input_dim, args.hidden_dim, num_classes,
                               num_layers=args.num_layers, dropout=args.dropout).to(device)
    elif args.model == "mlp":
        model = DeepMLP(input_dim, num_classes).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
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
        print(f"Epoch {epoch+1}: Train BalAcc={train_bal_acc:.4f} | "
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
    save_confusion_matrix(all_labels, all_preds, os.path.join(result_dir, "confusion_matrix.txt"))
    print(f"[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP/LSTM on single-frame features")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to 1-frame .pt file (e.g., features/pose_gaze_fm_1frame.pt)")
    parser.add_argument("--features", type=str, default="pose+gaze+fm")
    parser.add_argument("--model", type=str, default="mlp", choices=["lstm", "mlp"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5)

    args = parser.parse_args()
    args.features = parse_feature_selection(args.features)
    train(args)
