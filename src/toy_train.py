"""Phase 1 plumbing check only (PLAN.md Section 5, Phase 1).

Trains a linear head on a few hundred CIFAKE images to confirm the
end-to-end pipeline (load images -> frozen CLIP embeddings -> train head)
works and learns something above chance. This is NOT the real Phase 4
trainer (that trains on cached SID_Set embeddings per Section 3.3) —
this script exists only to satisfy Phase 1's acceptance criteria.
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features import extract_embeddings, load_backbone
from src.model import build_head


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_toy_paths(cifake_dir: str, n_per_class: int, split: str = "train"):
    real_dir = os.path.join(cifake_dir, split, "REAL")
    fake_dir = os.path.join(cifake_dir, split, "FAKE")
    real_files = sorted(os.listdir(real_dir))[:n_per_class]
    fake_files = sorted(os.listdir(fake_dir))[:n_per_class]
    paths = [os.path.join(real_dir, f) for f in real_files] + [
        os.path.join(fake_dir, f) for f in fake_files
    ]
    labels = [0] * len(real_files) + [1] * len(fake_files)  # 0=real, 1=fake
    return paths, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cifake-dir", required=True, help="Path to CIFAKE dataset root (contains train/ and test/)")
    parser.add_argument("--n-per-class", type=int, default=250, help="Images per class per split (toy scale)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoint-out", default="outputs/checkpoints/phase1_toy.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, preprocess, embedding_dim = load_backbone(
        cfg["backbone"]["family"], cfg["backbone"]["pretrained"], device
    )
    print(f"Backbone: {cfg['backbone']['family']}, embedding_dim (shape-checked): {embedding_dim}")

    train_paths, train_labels = collect_toy_paths(args.cifake_dir, args.n_per_class, split="train")
    test_paths, test_labels = collect_toy_paths(args.cifake_dir, max(50, args.n_per_class // 5), split="test")
    print(f"Train: {len(train_paths)} images, Test: {len(test_paths)} images")

    print("Extracting embeddings...")
    train_emb = extract_embeddings(train_paths, model, preprocess, device)
    test_emb = extract_embeddings(test_paths, model, preprocess, device)

    train_y = torch.tensor(train_labels, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.float32)

    adaptation_kwargs = {k: v for k, v in cfg["adaptation"].items() if k != "method"}
    head = build_head(cfg["adaptation"]["method"], embedding_dim, **adaptation_kwargs)
    head = head.to(device)
    train_emb, test_emb = train_emb.to(device), test_emb.to(device)
    train_y = train_y.to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        head.train()
        opt.zero_grad()
        logits = head(train_emb)
        loss = loss_fn(logits, train_y)
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                test_logits = head(test_emb)
                test_probs = torch.sigmoid(test_logits).cpu().numpy()
            test_preds = (test_probs >= cfg["eval"]["classification_threshold"]).astype(int)
            acc = accuracy_score(test_labels, test_preds)
            auc = roc_auc_score(test_labels, test_probs)
            print(f"Epoch {epoch+1}/{args.epochs}  loss={loss.item():.4f}  test_acc={acc:.4f}  test_auc={auc:.4f}")

    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "backbone_family": cfg["backbone"]["family"],
            "backbone_pretrained": cfg["backbone"]["pretrained"],
            "adaptation_method": cfg["adaptation"]["method"],
            "embedding_dim": embedding_dim,
        },
        args.checkpoint_out,
    )
    print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
