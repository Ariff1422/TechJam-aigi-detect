"""Phase 3 data pipeline (PLAN.md Section 2.2, Section 1.4, Phase 3).

Step 1 (extract_shards): read downloaded SID_Set parquet shards, drop the
"tampered" class (label==2, Section 0), write real (label==0) and
synthetic (label==1) images out to data/raw/{real,synthetic}/ as JPEGs,
subsampled to the configured target size.

Step 2 (check_resolution): tabulate native resolution real vs. synthetic
before any further processing (Section 1.4) — must be run and reviewed
before feature extraction, since a systematic gap would let the model
learn "resolution -> real" as a shortcut.

Step 3 (dedup_and_split): pHash near-duplicate grouping + stratified
70/15/15 train/val/test split (Section 2.2), written to data/splits/.
"""
import argparse
import io
import os
import random

import imagehash
import pandas as pd
import yaml
from PIL import Image

LABEL_REAL = 0
LABEL_SYNTHETIC = 1
LABEL_TAMPERED = 2


def extract_shards(shard_paths, out_dir, target_per_class, seed):
    random.seed(seed)
    real_dir = os.path.join(out_dir, "real")
    synth_dir = os.path.join(out_dir, "synthetic")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)

    real_count, synth_count, tampered_skipped = 0, 0, 0

    for shard_path in shard_paths:
        if real_count >= target_per_class and synth_count >= target_per_class:
            break
        df = pd.read_parquet(shard_path)
        for _, row in df.iterrows():
            label = row["label"]
            if label == LABEL_TAMPERED:
                tampered_skipped += 1
                continue
            if label == LABEL_REAL and real_count >= target_per_class:
                continue
            if label == LABEL_SYNTHETIC and synth_count >= target_per_class:
                continue

            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            if label == LABEL_REAL:
                out_path = os.path.join(real_dir, f"{row['img_id']}.jpg")
                real_count += 1
            else:
                out_path = os.path.join(synth_dir, f"{row['img_id']}.jpg")
                synth_count += 1
            img.save(out_path, format="JPEG", quality=95)

        print(f"After {os.path.basename(shard_path)}: real={real_count} synth={synth_count} tampered_skipped={tampered_skipped}")

    print(f"\nFinal: real={real_count}, synthetic={synth_count}, tampered_skipped={tampered_skipped}")
    return real_count, synth_count


def check_resolution(raw_dir):
    """PLAN.md Section 1.4 — tabulate native resolution real vs. synthetic
    BEFORE any resizing, to catch a resolution-based shortcut before it
    becomes an invisible Resize-transform robustness failure later."""
    rows = []
    for label_name in ["real", "synthetic"]:
        d = os.path.join(raw_dir, label_name)
        for fname in os.listdir(d):
            with Image.open(os.path.join(d, fname)) as img:
                w, h = img.size
            rows.append({"label": label_name, "width": w, "height": h, "megapixels": w * h / 1e6})
    df = pd.DataFrame(rows)
    summary = df.groupby("label")[["width", "height", "megapixels"]].describe()
    return df, summary


def normalize_resolution(raw_dir, common_size=512):
    """PLAN.md Section 1.4 mitigation — resize every image (both classes,
    identically) to a fixed common resolution in-place, so native
    resolution can never become a usable shortcut for either class.
    Run this AFTER check_resolution confirms a systematic gap exists, and
    BEFORE feature extraction (Section 3.3)."""
    counts = {"real": 0, "synthetic": 0}
    for label_name in ["real", "synthetic"]:
        d = os.path.join(raw_dir, label_name)
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            with Image.open(path) as img:
                img = img.convert("RGB")
                if img.size != (common_size, common_size):
                    img = img.resize((common_size, common_size), Image.BILINEAR)
                    img.save(path, format="JPEG", quality=95)
            counts[label_name] += 1
        print(f"{label_name}: normalized {counts[label_name]} images to {common_size}x{common_size}")
    return counts


def dedup_and_split(raw_dir, splits_dir, split_ratios, seed, hash_size=8, near_dup_threshold=4):
    """PLAN.md Section 2.2 — pHash near-duplicate grouping (not generator
    splitting, since SID_Set's synthetic side is single-generator/FLUX),
    then stratified 70/15/15 split with near-duplicate groups kept together."""
    random.seed(seed)

    items = []  # (path, label, phash)
    for label_name, label in [("real", 0), ("synthetic", 1)]:
        d = os.path.join(raw_dir, label_name)
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            with Image.open(path) as img:
                ph = imagehash.phash(img, hash_size=hash_size)
            items.append({"path": path, "label": label, "phash": ph})

    # Union-find style grouping: any two images within near_dup_threshold
    # Hamming distance land in the same group, so they can't be split apart.
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Bucket by hash prefix to avoid full O(n^2) comparison at n ~ 18K.
    from collections import defaultdict
    buckets = defaultdict(list)
    for idx, item in enumerate(items):
        buckets[str(item["phash"])[:4]].append(idx)

    for bucket_indices in buckets.values():
        for i in range(len(bucket_indices)):
            for j in range(i + 1, len(bucket_indices)):
                a, b = bucket_indices[i], bucket_indices[j]
                if items[a]["phash"] - items[b]["phash"] <= near_dup_threshold:
                    union(a, b)

    groups = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    group_list = list(groups.values())
    random.shuffle(group_list)

    n_dup_groups = sum(1 for g in group_list if len(g) > 1)
    n_dup_images = sum(len(g) for g in group_list if len(g) > 1)
    print(f"Near-duplicate groups found: {n_dup_groups} groups covering {n_dup_images} images (threshold={near_dup_threshold})")

    # Assign whole groups to splits, tracking running counts to keep
    # roughly stratified 50:50 real:fake in each split.
    target_counts = {
        split: {"real": 0, "synthetic": 0}
        for split in split_ratios
    }
    total_real = sum(1 for it in items if it["label"] == 0)
    total_synth = sum(1 for it in items if it["label"] == 1)
    for split, ratio in split_ratios.items():
        target_counts[split]["real"] = ratio * total_real
        target_counts[split]["synthetic"] = ratio * total_synth

    current_counts = {split: {"real": 0, "synthetic": 0} for split in split_ratios}
    assignment = {}  # idx -> split

    def deficit(split):
        r = target_counts[split]["real"] - current_counts[split]["real"]
        s = target_counts[split]["synthetic"] - current_counts[split]["synthetic"]
        return r + s

    for group in group_list:
        group_real = sum(1 for idx in group if items[idx]["label"] == 0)
        group_synth = sum(1 for idx in group if items[idx]["label"] == 1)
        best_split = max(split_ratios.keys(), key=deficit)
        for idx in group:
            assignment[idx] = best_split
        current_counts[best_split]["real"] += group_real
        current_counts[best_split]["synthetic"] += group_synth

    # Write order only, NOT split membership: writing in `items`' original
    # construction order (all real, then all synthetic, since that's how
    # the extraction loop built it) left every split file all-real-then-
    # all-synthetic, which is harmless for code that shuffles before
    # reading but a latent risk for anything that reads positionally.
    # Shuffle each split's row order before writing so files are properly
    # interleaved; `assignment` (which image landed in which split) is
    # untouched by this.
    os.makedirs(splits_dir, exist_ok=True)
    rows_by_split = {split: [] for split in split_ratios}
    for idx, item in enumerate(items):
        split = assignment[idx]
        rows_by_split[split].append(f"{item['path']}\t{item['label']}\n")

    for split, rows in rows_by_split.items():
        random.shuffle(rows)
        with open(os.path.join(splits_dir, f"{split}.txt"), "w") as f:
            f.writelines(rows)

    for split in split_ratios:
        c = current_counts[split]
        print(f"{split}: real={int(c['real'])} synthetic={int(c['synthetic'])} total={int(c['real']+c['synthetic'])}")

    return current_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--step", required=True, choices=["extract", "check_resolution", "normalize_resolution", "split"])
    parser.add_argument("--common-size", type=int, default=512, help="Target square size for normalize_resolution")
    parser.add_argument("--shard-dir", help="Directory containing downloaded SID_Set parquet shards (for --step extract)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["paths"]["raw_data_dir"]
    splits_dir = cfg["paths"]["splits_dir"]
    seed = cfg["seed"]

    if args.step == "extract":
        shard_paths = sorted(
            os.path.join(args.shard_dir, f) for f in os.listdir(args.shard_dir) if f.endswith(".parquet")
        )
        target_per_class = cfg["data"]["subsample_size"] // 2
        # Buffer above the final target so dedup filtering doesn't leave us short.
        extract_shards(shard_paths, raw_dir, int(target_per_class * 1.05), seed)

    elif args.step == "check_resolution":
        df, summary = check_resolution(raw_dir)
        print(summary)
        df.to_csv(os.path.join(raw_dir, "resolution_check.csv"), index=False)

    elif args.step == "normalize_resolution":
        normalize_resolution(raw_dir, args.common_size)

    elif args.step == "split":
        split_ratios = cfg["data"]["split"]
        dedup_and_split(raw_dir, splits_dir, split_ratios, seed)


if __name__ == "__main__":
    main()
