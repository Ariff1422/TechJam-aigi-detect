"""One-time quick-robustness cache (PLAN.md Phase 4's per-epoch robustness
tracking requirement).

Takes a small, fixed subsample of the VAL split (genuinely held-out —
never trained on), applies a handful of held-in transforms, embeds them
once through the frozen backbone, and saves the result to its own small
file, separate from the main val cache (which is clean-only per Section
3.3). train.py loads this once at startup and does pure array indexing
per epoch — no live transform, no backbone call during training, while
still measuring genuinely held-out robustness rather than a train-set
proxy.

Small enough (a few hundred images x a few tags) to run in seconds, so
it doesn't need cache_embeddings.py's shard/resume machinery.
"""
import argparse
import os
import random

import numpy as np
import yaml
from PIL import Image

from src.features import extract_embeddings_from_images, load_backbone
from src.transforms import apply_transform

QUICK_ROBUSTNESS_TRANSFORMS = [("jpeg", 50), ("gaussian_blur", 1.0), ("center_crop", 0.8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                         help="Which held-out split to subsample from (default val)")
    parser.add_argument("--n-samples", type=int, default=250)
    parser.add_argument("--out", default="data/cache/quick_robustness_val.npz")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rng = random.Random(cfg["seed"])

    split_path = os.path.join(cfg["paths"]["splits_dir"], f"{args.split}.txt")
    paths, labels = [], []
    with open(split_path) as f:
        for line in f:
            p, l = line.strip().split("\t")
            paths.append(p)
            labels.append(int(l))

    idx = list(range(len(paths)))
    rng.shuffle(idx)
    idx = idx[: args.n_samples]
    sub_paths = [paths[i] for i in idx]
    sub_labels = [labels[i] for i in idx]
    print(f"Subsampled {len(sub_paths)} images from {args.split} split "
          f"(seed={cfg['seed']}) for quick robustness caching")

    device = "cpu"
    model, preprocess, embedding_dim = load_backbone(
        cfg["backbone"]["family"], cfg["backbone"]["pretrained"], device
    )

    source_images = [Image.open(p).convert("RGB") for p in sub_paths]

    all_embeddings, all_labels, all_tags = [], [], []

    # clean, so the check can also report a same-subsample clean baseline
    emb = extract_embeddings_from_images(source_images, model, preprocess, device)
    all_embeddings.append(emb.numpy())
    all_labels.extend(sub_labels)
    all_tags.extend(["clean"] * len(sub_paths))

    for name, severity in QUICK_ROBUSTNESS_TRANSFORMS:
        tag = f"{name}_{severity}"
        transformed = [apply_transform(img, name, severity) for img in source_images]
        emb = extract_embeddings_from_images(transformed, model, preprocess, device)
        all_embeddings.append(emb.numpy())
        all_labels.extend(sub_labels)
        all_tags.extend([tag] * len(sub_paths))
        print(f"  cached tag={tag}: {emb.shape[0]} embeddings")

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    tags = np.array(all_tags)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out + ".tmp.npz", embeddings=embeddings, labels=labels, tags=tags)
    os.replace(args.out + ".tmp.npz", args.out)
    print(f"Saved {embeddings.shape[0]} total embeddings ({len(set(tags.tolist()))} tags) to {args.out}")


if __name__ == "__main__":
    main()
