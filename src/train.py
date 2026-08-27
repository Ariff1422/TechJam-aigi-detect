"""Phase 4 real trainer (PLAN.md Section 5, Phase 4).

Trains the classifier head on cached embeddings produced by
cache_embeddings.py (Section 3.3) — NOT toy_train.py's from-scratch
CIFAKE plumbing check. Hyperparameters, early stopping, and the
classification threshold all follow Phase 4's spec exactly:
AdamW, lr=1e-3, weight_decay=1e-4, batch size 256, up to 50 epochs,
early stopping on validation AUC, fixed random seed, threshold=0.5
unless calibrated and documented otherwise (Section 6).

The per-epoch "quick robustness check" loads a small, separately-cached
file (produced once by cache_quick_robustness.py) containing a held-out
val subsample under a few transforms, embedded once through the frozen
backbone ahead of time. It is loaded once at startup and then indexed
with plain arrays every epoch — never a live PIL transform + fresh CLIP
forward pass during training, which would defeat the point of caching,
and never train-set embeddings, which would only measure fit under
augmentation rather than genuine held-out generalization.
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.model import build_head


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cached_split(cache_dir, split_name):
    """Concatenate all shard .npz files for a split into one embeddings/labels pair."""
    shard_files = sorted(
        f for f in os.listdir(cache_dir) if f.startswith(f"{split_name}_shard") and f.endswith(".npz")
    )
    if not shard_files:
        raise FileNotFoundError(f"No cached shards found for split={split_name} in {cache_dir}")

    all_emb, all_labels, all_tags = [], [], []
    for fname in shard_files:
        d = np.load(os.path.join(cache_dir, fname), allow_pickle=True)
        all_emb.append(d["embeddings"])
        all_labels.append(d["labels"])
        all_tags.append(d["tags"])
    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    tags = np.concatenate(all_tags, axis=0)
    return embeddings, labels, tags


def quick_robustness_check(head, tagged_emb_by_tag, device, threshold):
    """Phase 4's 'quick 2-3 transform robustness check', run every epoch
    from cached tagged embeddings only (Section 3.3) — NOT the full grid
    (that's robustness_eval.py, Phase 2/5), and NOT a live transform +
    fresh CLIP pass. Returns {tag: (acc, auc)}."""
    results = {}
    for tag, (emb, labels) in tagged_emb_by_tag.items():
        acc, auc, _ = evaluate(head, emb, labels, device, threshold)
        results[tag] = (acc, auc)
    return results


def evaluate(head, embeddings, labels, device, threshold, batch_size=1024):
    head.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i : i + batch_size]).float().to(device)
            logits = head(batch)
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs_all)
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    return acc, auc, probs


def load_quick_robustness_cache(path):
    """Loads the small, separately-cached, genuinely held-out quick
    robustness file produced once by cache_quick_robustness.py. Returns
    {tag: (embeddings, labels)}. Pure array indexing at train time —
    no image loading, no transform, no backbone call."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Quick robustness cache not found at {path}. Run "
            f"`python -m src.cache_quick_robustness` first (one-time, a few seconds)."
        )
    d = np.load(path, allow_pickle=True)
    embeddings, labels, tags = d["embeddings"], d["labels"], d["tags"]
    result = {}
    for tag in sorted(set(tags.tolist())):
        mask = tags == tag
        result[tag] = (embeddings[mask], labels[mask])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--checkpoint-out", default="outputs/checkpoints/phase4_model.pt")
    parser.add_argument("--quick-robustness-cache", default="data/cache/quick_robustness_val.npz")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading cached embeddings...")
    train_emb, train_labels, train_tags = load_cached_split(args.cache_dir, "train")
    val_emb, val_labels, val_tags = load_cached_split(args.cache_dir, "val")
    print(f"Train: {train_emb.shape[0]} cached versions ({len(set(train_tags))} distinct tags)")
    print(f"Val: {val_emb.shape[0]} cached versions")

    embedding_dim = train_emb.shape[1]
    assert embedding_dim == cfg["backbone"]["embedding_dim"], (
        f"Cached embedding dim {embedding_dim} != config's declared "
        f"backbone.embedding_dim {cfg['backbone']['embedding_dim']}"
    )

    adaptation_kwargs = {k: v for k, v in cfg["adaptation"].items() if k != "method"}
    head = build_head(cfg["adaptation"]["method"], embedding_dim, **adaptation_kwargs).to(device)

    # Quick per-epoch robustness check, loaded once from a small,
    # genuinely held-out (val-subsample) cache — no live transform, no
    # backbone call during training.
    quick_robustness_cache = load_quick_robustness_cache(args.quick_robustness_cache)
    quick_tags = sorted(quick_robustness_cache.keys())
    print(f"Quick robustness check tags: {quick_tags} "
          f"({[quick_robustness_cache[t][0].shape[0] for t in quick_tags]} cached embeddings each, held-out val subsample)")

    train_cfg = cfg["train"]
    opt = torch.optim.AdamW(head.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(train_emb).float(), torch.from_numpy(train_labels).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)

    threshold = cfg["eval"]["classification_threshold"]
    patience = train_cfg["early_stopping_patience"]
    best_val_auc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(train_cfg["max_epochs"]):
        head.train()
        epoch_loss = 0.0
        for batch_emb, batch_labels in train_loader:
            batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)
            opt.zero_grad()
            logits = head(batch_emb)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch_emb.size(0)
        epoch_loss /= len(train_dataset)

        val_acc, val_auc, _ = evaluate(head, val_emb, val_labels, device, threshold)
        robustness = quick_robustness_check(head, quick_robustness_cache, device, threshold)
        robustness_str = "  ".join(f"{k}: acc={v[0]:.3f}/auc={v[1]:.3f}" for k, v in robustness.items())
        print(f"Epoch {epoch+1}/{train_cfg['max_epochs']}  train_loss={epoch_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_auc={val_auc:.4f}  |  {robustness_str}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (no val_auc improvement for {patience} epochs)")
                break

    head.load_state_dict(best_state)
    final_val_acc, final_val_auc, _ = evaluate(head, val_emb, val_labels, device, threshold)
    print(f"\nBest checkpoint: val_acc={final_val_acc:.4f}  val_auc={final_val_auc:.4f}  "
          f"(threshold={threshold})")

    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "backbone_family": cfg["backbone"]["family"],
            "backbone_pretrained": cfg["backbone"]["pretrained"],
            "adaptation_method": cfg["adaptation"]["method"],
            "embedding_dim": embedding_dim,
            "classification_threshold": threshold,
            "seed": cfg["seed"],
            "best_val_auc": best_val_auc,
        },
        args.checkpoint_out,
    )
    print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
