"""Feature-caching step (PLAN.md Section 3.3, Phase 3).

For each image in a split, generate the held-in (transform, severity)
versions + 1 clean version (train split) or just the clean version
(val/test splits), embed all of them through the frozen backbone once,
and cache the embeddings to disk.

Writes incrementally in shards (default 500 source images per shard) so
a crash or interrupt partway through only loses the current shard's
progress, not the whole run, and a re-run skips shards already written
to disk (resumable) rather than redoing completed work.
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml
from PIL import Image

from src.features import extract_embeddings_from_images, load_backbone
from src.transforms import HELD_IN_TRANSFORMS, TRANSFORM_GRID, apply_transform


def load_split(split_path):
    paths, labels = [], []
    with open(split_path) as f:
        for line in f:
            path, label = line.strip().split("\t")
            paths.append(path)
            labels.append(int(label))
    return paths, labels


def build_versions(image, include_augmented: bool):
    """Returns list of (version_tag, PIL.Image). version_tag='clean' for
    the unmodified image, else '{transform}_{severity}'."""
    versions = [("clean", image)]
    if include_augmented:
        for name in HELD_IN_TRANSFORMS:
            for sev in TRANSFORM_GRID[name]:
                versions.append((f"{name}_{sev}", apply_transform(image, name, sev)))
    return versions


def cache_split(split_name, paths, labels, out_dir, model, preprocess, device,
                 include_augmented, shard_size=500, batch_size=32):
    os.makedirs(out_dir, exist_ok=True)
    n_shards = (len(paths) + shard_size - 1) // shard_size

    t_start = time.time()
    for shard_idx in range(n_shards):
        shard_path = os.path.join(out_dir, f"{split_name}_shard{shard_idx:04d}.npz")
        if os.path.exists(shard_path):
            print(f"[{split_name}] shard {shard_idx+1}/{n_shards} already cached, skipping")
            continue

        s, e = shard_idx * shard_size, min((shard_idx + 1) * shard_size, len(paths))
        shard_paths = paths[s:e]
        shard_labels = labels[s:e]

        all_embeddings, all_labels, all_tags, all_src_paths = [], [], [], []
        for path, label in zip(shard_paths, shard_labels):
            img = Image.open(path).convert("RGB")
            versions = build_versions(img, include_augmented)
            imgs = [v[1] for v in versions]
            tags = [v[0] for v in versions]
            emb = extract_embeddings_from_images(imgs, model, preprocess, device, batch_size)
            all_embeddings.append(emb.numpy())
            all_labels.extend([label] * len(versions))
            all_tags.extend(tags)
            all_src_paths.extend([path] * len(versions))

        embeddings = np.concatenate(all_embeddings, axis=0)
        np.savez_compressed(
            shard_path + ".tmp.npz",
            embeddings=embeddings,
            labels=np.array(all_labels, dtype=np.int64),
            tags=np.array(all_tags),
            src_paths=np.array(all_src_paths),
        )
        # Atomic-ish: rename only after the write fully succeeds, so a crash
        # mid-write never leaves a shard file that looks done but isn't.
        os.replace(shard_path + ".tmp.npz", shard_path)

        elapsed = time.time() - t_start
        print(f"[{split_name}] shard {shard_idx+1}/{n_shards} done "
              f"({len(shard_paths)} images -> {embeddings.shape[0]} versions), "
              f"elapsed {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--shard-size", type=int, default=500, help="Source images per shard file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, embedding_dim = load_backbone(
        cfg["backbone"]["family"], cfg["backbone"]["pretrained"], device
    )
    print(f"Backbone: {cfg['backbone']['family']}, embedding_dim={embedding_dim}, device={device}")

    split_path = os.path.join(cfg["paths"]["splits_dir"], f"{args.split}.txt")
    paths, labels = load_split(split_path)
    print(f"Loaded {len(paths)} images for split={args.split}")

    include_augmented = args.split == "train"  # Section 3.3: only train gets the 9 augmented + 1 clean versions
    cache_split(
        args.split, paths, labels, args.cache_dir, model, preprocess, device,
        include_augmented, args.shard_size, args.batch_size,
    )


if __name__ == "__main__":
    main()
