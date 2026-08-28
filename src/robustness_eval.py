"""Robustness evaluation harness (PLAN.md Section 3.1/3.4, Phase 2/5).

Given a model checkpoint (or --random-model for the harness sanity check)
and a labeled image folder (subfolders REAL/ and FAKE/, matching the
CIFAKE/SID_Set convention used elsewhere in this repo), computes and
caches embeddings for every transform x severity combination from
Section 3.1 plus the Section 3.4 stacked/combined worst-case tests, then
scores them into an accuracy/AUC table, one row per cell.

Embeddings are cached per-condition to disk (one .npz per cell, atomic
write, skip-if-exists resume) rather than held in memory for one long
single-shot run that writes nothing until the very end — the earlier
design lost all progress to an external kill partway through a ~40min
run, twice. Now a kill loses at most one condition's worth of work, and
re-scoring at a different threshold (e.g. comparing 0.5 vs. a calibrated
value) is pure array math against the cache, not a rerun of the backbone.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

from src.checkpoint import load_model_from_checkpoint
from src.features import extract_embeddings_from_images, load_backbone
from src.model import build_head
from src.transforms import HELD_OUT_TRANSFORMS, TRANSFORM_GRID, apply_chain, apply_transform

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


def condition_list():
    """Returns [(cell_name, severity_label, held_out, transform_fn_or_None)],
    where transform_fn_or_None(img) -> img; None means clean (no transform)."""
    conditions = [("clean", "n/a", False, None)]
    for transform_name, severities in TRANSFORM_GRID.items():
        held_out = transform_name in HELD_OUT_TRANSFORMS
        for severity in severities:
            conditions.append((
                transform_name, str(severity), held_out,
                lambda img, n=transform_name, s=severity: apply_transform(img, n, s),
            ))
    for chain_name, steps in STACKED_CHAINS.items():
        conditions.append((
            f"STACKED:{chain_name}", "+".join(f"{n}={s}" for n, s in steps), False,
            lambda img, st=steps: apply_chain(img, st),
        ))
    return conditions


def cache_condition_embeddings(cache_dir, source_images, labels, model, preprocess, device, batch_size=32):
    """Computes and caches one .npz per condition (transform/severity
    cell), skipping any cell already cached. Returns nothing — read back
    with load_cached_condition_embeddings for scoring."""
    os.makedirs(cache_dir, exist_ok=True)
    conditions = condition_list()

    for cell_name, severity_label, held_out, transform_fn in conditions:
        # Sanitize for use in a filename — cell names contain ':' (STACKED:...)
        # and severity labels contain '/' (n/a) or '+'/'=' (stacked chain steps).
        safe_name = cell_name.replace(":", "_")
        safe_severity = severity_label.replace("/", "-").replace("=", "-").replace("+", "_")
        cache_path = os.path.join(cache_dir, f"{safe_name}__{safe_severity}.npz")
        if os.path.exists(cache_path):
            print(f"[{cell_name} sev={severity_label}] already cached, skipping")
            continue

        images = source_images if transform_fn is None else [transform_fn(img) for img in source_images]
        embeddings = extract_embeddings_from_images(images, model, preprocess, device, batch_size).numpy()

        np.savez_compressed(
            cache_path + ".tmp.npz",
            embeddings=embeddings,
            labels=np.array(labels, dtype=np.int64),
            transform=cell_name,
            severity=severity_label,
            held_out=held_out,
        )
        os.replace(cache_path + ".tmp.npz", cache_path)
        print(f"[{cell_name} sev={severity_label}] cached ({embeddings.shape[0]} embeddings)")


def score_cached_conditions(cache_dir, head, device, threshold):
    """Reads back every cached condition and scores it at the given
    threshold — pure array math, no backbone calls."""
    rows = []
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".npz"):
            continue
        d = np.load(os.path.join(cache_dir, fname), allow_pickle=True)
        embeddings = torch.from_numpy(d["embeddings"]).float().to(device)
        labels = d["labels"]
        with torch.no_grad():
            logits = head(embeddings)
            probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")
        rows.append({
            "transform": str(d["transform"]),
            "severity": str(d["severity"]),
            "held_out": bool(d["held_out"]),
            "accuracy": acc,
            "auc": auc,
        })
    return pd.DataFrame(rows)


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
    parser.add_argument("--embedding-cache-dir", default=None,
                         help="Where to cache per-condition embeddings (resumable). Defaults to a dir next to --output.")
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

    cache_dir = args.embedding_cache_dir or (os.path.splitext(args.output)[0] + "_embcache")

    print("Loading source images into memory...")
    source_images = [Image.open(p).convert("RGB") for p in paths]

    cache_condition_embeddings(cache_dir, source_images, labels, model, preprocess, device)

    df = score_cached_conditions(cache_dir, head, device, args.threshold)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)}-row robustness table to {args.output} (threshold={args.threshold})")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
