"""Phase B: heuristic-routed calibration test.

Builds a 2-bin router (clean-ish vs. degraded) from cheap pixel
heuristics (src/quality_heuristics.py), fits a threshold per bin on the
VAL split (never touching test), then evaluates fixed-0.5 vs.
heuristic-routed calibration on a realistic MIXED-CONDITION test set —
each image gets an unknown (to the pipeline) random distortion or none,
simulating real deployment where the true condition isn't known.

This is still not a deployable production mechanism as shipped (2 bins,
one boolean rule) but it's no longer an oracle: the routing decision
uses only pixel-level heuristics computed from the image itself, not
ground-truth knowledge of which transform was applied.
"""
import argparse
import random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score

from src.checkpoint import load_model_from_checkpoint
from src.quality_heuristics import compute_all
from src.transforms import HELD_IN_TRANSFORMS, HELD_OUT_TRANSFORMS, TRANSFORM_GRID, apply_transform

ALL_TRANSFORMS = list(TRANSFORM_GRID.keys())


def load_split(split_path):
    paths, labels = [], []
    with open(split_path) as f:
        for line in f:
            p, l = line.strip().split("\t")
            paths.append(p)
            labels.append(int(l))
    return paths, labels


def is_degraded_by_heuristics(heuristics: dict, clean_percentiles: dict):
    """Percentile-based router: an image is 'degraded' if its blur_score
    falls below the clean distribution's low percentile (blurred/
    resized), OR its noise_score/jpeg_block_score exceed the clean
    distribution's high percentile (noisy/compressed). Percentile-based
    rather than mean-ratio because clean-image heuristic scores have
    very high natural variance (e.g. blur_score std ~= its own mean on
    real photos), so a fixed ratio-off-the-mean rule misclassifies a
    large fraction of genuinely clean images."""
    is_blurred = heuristics["blur_score"] < clean_percentiles["blur_p10"]
    is_noisy = heuristics["noise_score"] > clean_percentiles["noise_p90"]
    is_jpeg = heuristics["jpeg_block_score"] > clean_percentiles["jpeg_p90"]
    return is_blurred or is_noisy or is_jpeg


def best_threshold(probs, labels):
    best_acc, best_t = 0.0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        acc = accuracy_score(labels, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def build_mixed_test_set(paths, labels, seed, n):
    """Each image gets a random distortion (including 'clean') applied,
    unknown to the router/model at evaluation time — matches real
    deployment where you don't know what happened to an incoming image."""
    rng = random.Random(seed)
    idx = list(range(len(paths)))
    rng.shuffle(idx)
    idx = idx[:n]

    mixed = []
    for i in idx:
        choice = rng.random()
        if choice < 0.3:
            transform, severity = None, None
        else:
            transform = rng.choice(ALL_TRANSFORMS)
            severity = rng.choice(TRANSFORM_GRID[transform])
        mixed.append((paths[i], labels[i], transform, severity))
    return mixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/phase4_model.pt")
    parser.add_argument("--n-mixed", type=int, default=400)
    parser.add_argument("--n-calib", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cpu"
    model, preprocess, head, embedding_dim = load_model_from_checkpoint(args.checkpoint, device)
    head.eval()

    # --- Step 1: build clean-image heuristic reference + degraded-bin threshold from VAL split ---
    val_paths, val_labels = load_split("data/splits/val.txt")
    rng_calib = random.Random(123)  # different seed from the mixed-test-set build, no image overlap risk in practice negligible but kept distinct
    idx = list(range(len(val_paths)))
    rng_calib.shuffle(idx)
    calib_idx = idx[: args.n_calib]

    clean_heuristics = []
    clean_probs, clean_labels_arr = [], []
    degraded_records = []  # (heuristics, prob, label)

    print("Building calibration reference from val split...")
    for i in calib_idx:
        img = Image.open(val_paths[i]).convert("RGB")
        h_clean = compute_all(img)
        clean_heuristics.append(h_clean)

        with torch.no_grad():
            emb = model.encode_image(preprocess(img).unsqueeze(0))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            prob_clean = torch.sigmoid(head(emb)).item()
        clean_probs.append(prob_clean)
        clean_labels_arr.append(val_labels[i])

        # also sample one random held-in-or-out degradation of the SAME image for the degraded bin's threshold
        transform = random.Random(i).choice(ALL_TRANSFORMS)
        severity = random.Random(i + 1).choice(TRANSFORM_GRID[transform])
        deg_img = apply_transform(img, transform, severity)
        h_deg = compute_all(deg_img)
        with torch.no_grad():
            emb = model.encode_image(preprocess(deg_img).unsqueeze(0))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            prob_deg = torch.sigmoid(head(emb)).item()
        degraded_records.append((h_deg, prob_deg, val_labels[i]))

    blur_vals = [h["blur_score"] for h in clean_heuristics]
    noise_vals = [h["noise_score"] for h in clean_heuristics]
    jpeg_vals = [h["jpeg_block_score"] for h in clean_heuristics]
    clean_percentiles = {
        "blur_p10": float(np.percentile(blur_vals, 10)),
        "noise_p90": float(np.percentile(noise_vals, 90)),
        "jpeg_p90": float(np.percentile(jpeg_vals, 90)),
    }
    print(f"Clean-distribution percentiles (val, n={len(clean_heuristics)}): {clean_percentiles}")

    clean_probs = np.array(clean_probs)
    clean_labels_arr = np.array(clean_labels_arr)
    clean_t, _ = best_threshold(clean_probs, clean_labels_arr)

    degraded_probs = np.array([r[1] for r in degraded_records])
    degraded_labels_arr = np.array([r[2] for r in degraded_records])
    degraded_t, _ = best_threshold(degraded_probs, degraded_labels_arr)

    print(f"Fitted thresholds from val split: clean-bin={clean_t:.2f}, degraded-bin={degraded_t:.2f}")

    # --- Step 2: build the mixed-condition test set from the TEST split ---
    test_paths, test_labels = load_split("data/splits/test.txt")
    mixed = build_mixed_test_set(test_paths, test_labels, args.seed, args.n_mixed)
    print(f"Built mixed test set: {len(mixed)} images, random unknown-to-router distortion each")

    fixed_preds, routed_preds, true_labels = [], [], []
    n_routed_degraded = 0

    for path, label, transform, severity in mixed:
        img = Image.open(path).convert("RGB")
        if transform is not None:
            img = apply_transform(img, transform, severity)

        h = compute_all(img)
        with torch.no_grad():
            emb = model.encode_image(preprocess(img).unsqueeze(0))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            prob = torch.sigmoid(head(emb)).item()

        fixed_preds.append(int(prob >= 0.5))

        routed_to_degraded = is_degraded_by_heuristics(h, clean_percentiles)
        chosen_t = degraded_t if routed_to_degraded else clean_t
        routed_preds.append(int(prob >= chosen_t))
        n_routed_degraded += int(routed_to_degraded)

        true_labels.append(label)

    acc_fixed = accuracy_score(true_labels, fixed_preds)
    acc_routed = accuracy_score(true_labels, routed_preds)

    print(f"\nMixed test set (n={len(mixed)}, {n_routed_degraded} routed to degraded bin):")
    print(f"  Fixed threshold 0.5:        accuracy = {acc_fixed:.4f}")
    print(f"  Heuristic-routed threshold: accuracy = {acc_routed:.4f}")
    print(f"  Gain: {acc_routed - acc_fixed:+.4f}")


if __name__ == "__main__":
    main()
