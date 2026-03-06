"""
Convert single-frame features into temporal sequences via sliding window
with uniform subsampling.

Input:  (X: [N_frames, 48], y: [N_frames])  — from merge_features.py
Output: (X: [N_seq, sample_frames, 48], y: [N_seq])
"""

import os
import argparse
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm


def most_common_label(tensor_1d):
    """Return the most frequent label in a 1D tensor."""
    return Counter(tensor_1d.tolist()).most_common(1)[0][0]


def uniform_subsample_indices(span_len, sample_len):
    """Return uniformly spaced indices over [0, span_len-1].
    Always includes the first and last frame when sample_len >= 2.
    """
    if sample_len == 1:
        return np.array([span_len // 2], dtype=int)
    return np.linspace(0, span_len - 1, num=sample_len, dtype=int)


def generate_subsampled_sequences(data_path, span_frames, sample_frames, step,
                                  save_path, label_from="span"):
    """Slide a span_frames-wide window; uniformly subsample sample_frames from it.

    Args:
        label_from: How to derive the sequence label.
            - "span":   majority vote over ALL frames in the span (recommended)
            - "sample": majority vote over only the sampled frames
            - "center": label of the temporal center frame
    """
    print(f"[INFO] Loading data from: {data_path}")
    X, y = torch.load(data_path)
    print(f"[INFO] Loaded data: X={X.shape}, y={y.shape}")

    assert X.ndim == 2, f"Expected X shape [N, 48], got {X.shape}"
    assert y.ndim == 1 and len(y) == len(X)
    assert 1 <= sample_frames <= span_frames

    idx_map = uniform_subsample_indices(span_frames, sample_frames)
    sequences, seq_labels = [], []

    for start in tqdm(range(0, len(X) - span_frames + 1, step), desc="Generating sequences"):
        end = start + span_frames
        span_X = X[start:end]
        span_y = y[start:end]

        sampled_X = span_X[idx_map]

        if label_from == "span":
            label = most_common_label(span_y)
        elif label_from == "sample":
            label = most_common_label(span_y[idx_map])
        elif label_from == "center":
            label = span_y[span_frames // 2].item()
        else:
            raise ValueError(f"Unknown label_from='{label_from}'")

        sequences.append(sampled_X)
        seq_labels.append(label)

    X_seq = torch.stack(sequences)
    y_seq = torch.tensor(seq_labels)

    print(f"[INFO] Created {X_seq.shape[0]} sequences of length {sample_frames} "
          f"(from spans of {span_frames}).")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save((X_seq, y_seq), save_path)
    print(f"[DONE] Saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate temporal sequences from 1-frame features")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to 1-frame feature .pt file")
    parser.add_argument("--span_frames", type=int, default=30,
                        help="Temporal span size (e.g., 30 frames = 3s at 10fps)")
    parser.add_argument("--sample_frames", type=int, default=10,
                        help="Frames to uniformly subsample from each span")
    parser.add_argument("--step", type=int, default=10,
                        help="Sliding stride between spans")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output path for sequence .pt file")
    parser.add_argument("--label_from", type=str, default="span",
                        choices=["span", "sample", "center"],
                        help="Label assignment strategy (default: span)")
    args = parser.parse_args()

    generate_subsampled_sequences(
        data_path=args.data_path,
        span_frames=args.span_frames,
        sample_frames=args.sample_frames,
        step=args.step,
        save_path=args.save_path,
        label_from=args.label_from,
    )
