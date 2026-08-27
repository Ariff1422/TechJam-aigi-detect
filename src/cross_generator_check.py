"""Cross-generator generalization check (PLAN.md Section 7 item 5).

Evaluates a trained checkpoint (inference only — no gradient updates, no
fine-tuning) on a sample from a generator/real-source distribution
different from training (SID_Set: FLUX-generated, OpenImages-sourced
real photos). Reports AUC on the out-of-distribution sample to check
whether the model learned general synthesis cues or something narrower
(e.g. a file-format/compression artifact specific to SID_Set's two
sources).

Primary intended source (PLAN.md Section 2.1/7): the brief's own
WildFake COCO-vs-DALL·E "demonstration purposes only" subset — never
trained on, evaluation/inference only. Fallback: CIFAKE (Stable
Diffusion-based, CIFAR-10-sourced), a weaker but readily available
substitute along the same two axes (different generator, different
real-photo source).
"""
import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from src.checkpoint import load_model_from_checkpoint
from src.features import extract_embeddings

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_labeled_paths(real_dir, fake_dir, max_per_class=None):
    real_files = sorted(f for f in os.listdir(real_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    fake_files = sorted(f for f in os.listdir(fake_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    if max_per_class:
        real_files, fake_files = real_files[:max_per_class], fake_files[:max_per_class]
    paths = [os.path.join(real_dir, f) for f in real_files] + [os.path.join(fake_dir, f) for f in fake_files]
    labels = [0] * len(real_files) + [1] * len(fake_files)
    return paths, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/phase4_model.pt")
    parser.add_argument("--real-dir", required=True, help="Directory of real images from the OOD source")
    parser.add_argument("--fake-dir", required=True, help="Directory of AI-generated images from the OOD source")
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--source-name", default="CIFAKE", help="Label for this OOD source in the report")
    args = parser.parse_args()

    device = "cpu"
    model, preprocess, head, embedding_dim = load_model_from_checkpoint(args.checkpoint, device)

    paths, labels = collect_labeled_paths(args.real_dir, args.fake_dir, args.max_per_class)
    print(f"[{args.source_name}] Loaded {len(paths)} images "
          f"({labels.count(0)} real, {labels.count(1)} fake) — inference only, no gradient updates")

    embeddings = extract_embeddings(paths, model, preprocess, device)
    import torch
    with torch.no_grad():
        logits = head(embeddings)
        probs = torch.sigmoid(logits).numpy()

    preds = (probs >= args.threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    print(f"\n[{args.source_name}] Cross-generator check (out-of-distribution, checkpoint never trained on this data):")
    print(f"  accuracy={acc:.4f}  AUC={auc:.4f}  (threshold={args.threshold}, n={len(paths)})")


if __name__ == "__main__":
    main()
