"""Robustness evaluation harness (PLAN.md Section 3.1/3.4, Phase 2/5).

Given a model checkpoint (or --random-model for the harness sanity check)
and a labeled image folder (subfolders REAL/ and FAKE/, matching the
CIFAKE/SID_Set convention used elsewhere in this repo), runs every
transform x severity combination from Section 3.1 plus the Section 3.4
stacked/combined worst-case tests, and outputs an accuracy/AUC table,
one row per cell.
"""
import argparse
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

from src.checkpoint import load_model_from_checkpoint
from src.features import extract_embeddings_from_images, load_backbone
from src.model import build_head
from src.transforms import TRANSFORM_GRID, apply_chain, apply_transform

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Section 3.4 stacked/combined worst-case tests, beyond the required grid.
STACKED_CHAINS = {
    "resize0.25_blur2.0_jpeg30": [("resize", 0.25), ("gaussian_blur", 2.0), ("jpeg", 30)],
    "resize0.5_noise0.10_jpeg50": [("resize", 0.5), ("gaussian_noise", 0.10), ("jpeg", 50)],
}


def collect_labeled_paths(labeled_dir: str, max_per_class: int = None):
    """Expects labeled_dir/REAL/*, labeled_dir/FAKE/* (0=real, 1=fake,
    matching the convention locked in PLAN.md Section 0 / toy_train.py)."""
    real_dir = os.path.join(labeled_dir, "REAL")
    fake_dir = os.path.join(labeled_dir, "FAKE")
    real_files = sorted(f for f in os.listdir(real_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    fake_files = sorted(f for f in os.listdir(fake_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    if max_per_class:
        real_files, fake_files = real_files[:max_per_class], fake_files[:max_per_class]
    paths = [os.path.join(real_dir, f) for f in real_files] + [os.path.join(fake_dir, f) for f in fake_files]
    labels = [0] * len(real_files) + [1] * len(fake_files)
    return paths, labels


def evaluate_cell(images, labels, model, preprocess, head, device, threshold: float):
    embeddings = extract_embeddings_from_images(images, model, preprocess, device).to(device)
    with torch.no_grad():
        logits = head(embeddings)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")  # only one class present in labels
    return acc, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", required=True, help="Dir with REAL/ and FAKE/ subfolders")
    parser.add_argument("--checkpoint", help="Trained checkpoint (.pt). Omit and pass --random-model to sanity-check the harness itself.")
    parser.add_argument("--random-model", action="store_true", help="Use an untrained head with random weights instead of a checkpoint (Phase 2 acceptance criteria)")
    parser.add_argument("--backbone", default="clip_vitb32", help="Backbone family, only used with --random-model")
    parser.add_argument("--adaptation-method", default="frozen_linear", help="Head type, only used with --random-model")
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap images per class (speed vs. coverage tradeoff)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/robustness_table.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.random_model:
        model, preprocess, embedding_dim = load_backbone(args.backbone, "openai", device)
        head = build_head(args.adaptation_method, embedding_dim).to(device).eval()
        print(f"Using UNTRAINED random-weight head ({args.adaptation_method} on {args.backbone}) — harness sanity check mode.")
    else:
        if not args.checkpoint:
            raise ValueError("Must pass --checkpoint unless --random-model is set")
        model, preprocess, head, embedding_dim = load_model_from_checkpoint(args.checkpoint, device)

    paths, labels = collect_labeled_paths(args.labeled_dir, args.max_per_class)
    print(f"Loaded {len(paths)} labeled images ({sum(1 for l in labels if l == 0)} real, {sum(1 for l in labels if l == 1)} fake)")

    print("Loading source images into memory...")
    source_images = [Image.open(p).convert("RGB") for p in paths]

    rows = []

    # Clean baseline (no transform) — every table needs this reference row.
    acc, auc = evaluate_cell(source_images, labels, model, preprocess, head, device, args.threshold)
    rows.append({"transform": "clean", "severity": "n/a", "held_out": False, "accuracy": acc, "auc": auc})
    print(f"clean            acc={acc:.4f}  auc={auc:.4f}")

    from src.transforms import HELD_OUT_TRANSFORMS

    for transform_name, severities in TRANSFORM_GRID.items():
        held_out = transform_name in HELD_OUT_TRANSFORMS
        for severity in severities:
            transformed = [apply_transform(img, transform_name, severity) for img in source_images]
            acc, auc = evaluate_cell(transformed, labels, model, preprocess, head, device, args.threshold)
            rows.append({
                "transform": transform_name,
                "severity": severity,
                "held_out": held_out,
                "accuracy": acc,
                "auc": auc,
            })
            print(f"{transform_name:<15} sev={severity:<6} held_out={held_out!s:<5} acc={acc:.4f}  auc={auc:.4f}")

    # Section 3.4 — stacked/combined, beyond-spec, reported separately.
    for chain_name, steps in STACKED_CHAINS.items():
        transformed = [apply_chain(img, steps) for img in source_images]
        acc, auc = evaluate_cell(transformed, labels, model, preprocess, head, device, args.threshold)
        rows.append({
            "transform": f"STACKED:{chain_name}",
            "severity": "+".join(f"{n}={s}" for n, s in steps),
            "held_out": False,
            "accuracy": acc,
            "auc": auc,
        })
        print(f"STACKED:{chain_name:<30} acc={acc:.4f}  auc={auc:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)}-row robustness table to {args.output}")


if __name__ == "__main__":
    main()
