"""Oracle per-condition threshold calibration check (PLAN.md error_analysis.md
Section 3 — "the decisive test of the calibration hypothesis").

For each cached robustness condition, splits its images in half (fixed
seed), fits the accuracy-maximizing threshold on one half, and evaluates
on the OTHER half at that threshold — avoiding the circularity of tuning
and testing on identical data.

This is an ORACLE / UPPER-BOUND diagnostic, not a deployable fix:
infer.py has no way to know a test image's true distortion condition at
inference time, so it cannot select the matching per-condition
threshold. This script measures the ceiling calibration alone could
achieve, given that information — it does not produce anything shippable.
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.checkpoint import load_model_from_checkpoint
from src.cross_generator_check import collect_labeled_paths
from src.features import extract_embeddings


def best_threshold(probs, labels):
    best_acc, best_t = 0.0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        acc = accuracy_score(labels, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def split_half_oracle(probs: np.ndarray, labels: np.ndarray, rng: np.random.RandomState):
    """Fits on one half, evaluates on the other. Returns
    (tuned_threshold, acc_at_global_0.5_on_eval_half, acc_at_tuned_on_eval_half).

    Takes a shared RandomState advancing across calls (not a fresh
    RandomState(seed) per call) to match the original investigation's
    methodology exactly — matters only for exact reproducibility of the
    reported numbers, not for the finding itself."""
    n = len(labels)
    idx = rng.permutation(n)
    half = n // 2
    fit_idx, eval_idx = idx[:half], idx[half:]

    tuned_t, _ = best_threshold(probs[fit_idx], labels[fit_idx])
    acc_global = accuracy_score(labels[eval_idx], (probs[eval_idx] >= 0.5).astype(int))
    acc_tuned = accuracy_score(labels[eval_idx], (probs[eval_idx] >= tuned_t).astype(int))
    return tuned_t, acc_global, acc_tuned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/phase4_model.pt")
    parser.add_argument("--robustness-cache-dir", default="data/cache/robustness_embcache")
    parser.add_argument("--cifake-real-dir", default=None, help="Optional: also run the oracle check on CIFAKE")
    parser.add_argument("--cifake-fake-dir", default=None)
    parser.add_argument("--cifake-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cpu"
    model, preprocess, head, embedding_dim = load_model_from_checkpoint(args.checkpoint, device)
    head.eval()

    rng = np.random.RandomState(args.seed)  # shared, advances across every condition below

    print(f"{'condition':<35} {'severity':<40} {'n':>6} {'tuned_t':>8} {'acc@0.5':>9} {'acc@oracle':>11} {'gain':>8}")

    for fname in sorted(os.listdir(args.robustness_cache_dir)):
        if not fname.endswith(".npz"):
            continue
        d = np.load(os.path.join(args.robustness_cache_dir, fname), allow_pickle=True)
        emb = torch.from_numpy(d["embeddings"]).float()
        labels = d["labels"]
        with torch.no_grad():
            probs = torch.sigmoid(head(emb)).numpy()

        tuned_t, acc_global, acc_tuned = split_half_oracle(probs, labels, rng)
        print(f"{str(d['transform']):<35} {str(d['severity']):<40} {len(labels):>6} "
              f"{tuned_t:>8.2f} {acc_global:>9.4f} {acc_tuned:>11.4f} {acc_tuned - acc_global:>+8.4f}")

    if args.cifake_real_dir and args.cifake_fake_dir:
        paths, labels = collect_labeled_paths(args.cifake_real_dir, args.cifake_fake_dir, args.cifake_n)
        labels = np.array(labels)
        embeddings = extract_embeddings(paths, model, preprocess, device)
        with torch.no_grad():
            probs = torch.sigmoid(head(embeddings)).numpy()
        tuned_t, acc_global, acc_tuned = split_half_oracle(probs, labels, rng)
        print(f"{'CIFAKE (OOD)':<35} {'n/a':<40} {len(labels):>6} "
              f"{tuned_t:>8.2f} {acc_global:>9.4f} {acc_tuned:>11.4f} {acc_tuned - acc_global:>+8.4f}")


if __name__ == "__main__":
    main()
