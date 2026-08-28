"""Temperature scaling (PLAN.md error_analysis.md Section 5 item 1).

Fits a single scalar T on the validation split's logits to minimize
NLL, then divides logits by T before sigmoid at inference. Reshapes
the confidence distribution rather than moving a fixed threshold
through it — the more promising direction identified after the
global-threshold-shift experiment was tested and rejected.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from src.checkpoint import load_model_from_checkpoint
from src.train import load_cached_split


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter=200, lr=0.01):
    """Fits T to minimize BCE-with-logits NLL on (logits/T, labels)."""
    log_t = nn.Parameter(torch.zeros(1))  # T = exp(log_t), starts at T=1
    opt = torch.optim.Adam([log_t], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(max_iter):
        opt.zero_grad()
        t = torch.exp(log_t)
        loss = loss_fn(logits / t, labels)
        loss.backward()
        opt.step()

    return torch.exp(log_t).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/phase4_model.pt")
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()

    device = "cpu"
    model, preprocess, head, embedding_dim = load_model_from_checkpoint(args.checkpoint, device)
    head.eval()

    val_emb, val_labels, val_tags = load_cached_split(args.cache_dir, "val")

    with torch.no_grad():
        val_logits = head(torch.from_numpy(val_emb).float())

    T = fit_temperature(val_logits, torch.from_numpy(val_labels).float())
    print(f"Fitted temperature T = {T:.4f}")

    # Report NLL before/after on val, for a sanity check the fit actually helped calibration there.
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        nll_before = loss_fn(val_logits, torch.from_numpy(val_labels).float()).item()
        nll_after = loss_fn(val_logits / T, torch.from_numpy(val_labels).float()).item()
    print(f"Val NLL before: {nll_before:.4f}  after: {nll_after:.4f}")

    return T


if __name__ == "__main__":
    main()
