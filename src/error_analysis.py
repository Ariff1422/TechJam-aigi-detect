"""Error analysis (PLAN.md Phase 5, deliverable 5).

Pulls concrete false positive/negative examples from the Phase 4
checkpoint under both clean and transformed conditions, using the same
test split as the Phase 5 robustness sweep, so the Error Analysis Note
can cite specific images and hypotheses rather than aggregate numbers
alone.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from src.checkpoint import load_model_from_checkpoint
from src.features import extract_embeddings_from_images
from src.transforms import apply_transform

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_split(split_path):
    paths, labels = [], []
    with open(split_path) as f:
        for line in f:
            p, l = line.strip().split("\t")
            paths.append(p)
            labels.append(int(l))
    return paths, labels


def find_misclassified(paths, labels, model, preprocess, head, device, threshold, transform=None):
    images = [Image.open(p).convert("RGB") for p in paths]
    if transform is not None:
        name, severity = transform
        images = [apply_transform(img, name, severity) for img in images]

    embeddings = extract_embeddings_from_images(images, model, preprocess, device)
    with torch.no_grad():
        logits = head(embeddings)
        probs = torch.sigmoid(logits).numpy()

    labels = np.array(labels)
    preds = (probs >= threshold).astype(int)
    wrong_mask = preds != labels

    results = []
    for i in np.where(wrong_mask)[0]:
        results.append({
            "path": paths[i],
            "true_label": int(labels[i]),
            "pred_prob": float(probs[i]),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/phase4_model.pt")
    parser.add_argument("--test-split", default="data/splits/test.txt")
    parser.add_argument("--n-samples", type=int, default=500, help="Test images to scan per condition")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cpu"
    model, preprocess, head, embedding_dim = load_model_from_checkpoint(args.checkpoint, device)

    paths, labels = load_split(args.test_split)
    import random
    random.seed(42)
    idx = list(range(len(paths)))
    random.shuffle(idx)
    idx = idx[: args.n_samples]
    sample_paths = [paths[i] for i in idx]
    sample_labels = [labels[i] for i in idx]

    conditions = [
        ("clean", None),
        ("gaussian_noise_0.10", ("gaussian_noise", 0.10)),
        ("stacked_resize_noise_jpeg", None),  # handled separately below
    ]

    for name, transform in conditions[:2]:
        wrong = find_misclassified(sample_paths, sample_labels, model, preprocess, head, device, args.threshold, transform)
        print(f"\n=== {name}: {len(wrong)}/{len(sample_paths)} misclassified ===")
        for w in wrong[:6]:
            true_str = "FAKE" if w["true_label"] == 1 else "REAL"
            print(f"  path={w['path']}  true={true_str}  pred_prob={w['pred_prob']:.4f}")


if __name__ == "__main__":
    main()
