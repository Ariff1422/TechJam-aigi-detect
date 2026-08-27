"""Phase 4b — LoRA fine-tuning experiment (PLAN.md Section 4b/7 item 1).

Unlike Phase 4 (train.py), this trains gradients through the actual CLIP
backbone (via a small LoRA adapter on the last few transformer blocks'
out_proj + MLP linears — PyTorch's fused MultiheadAttention in_proj
isn't a plain nn.Linear, so it isn't LoRA-targetable via peft directly;
Section 4b's "last few transformer blocks" is satisfied on the
sub-modules that are). This means Phase 4's cached embeddings can't be
reused — every step re-runs the (now partially trainable) backbone on
raw images, making this a much longer, heavier job than Phase 4's
seconds-per-epoch linear probe on pre-cached vectors.

Given the interrupt history during Phase 3's caching job (killed twice
by the environment with no warning), this script checkpoints
periodically (every --checkpoint-every steps, not just at the end) and
resumes from the latest checkpoint automatically if one exists — so an
external kill loses at most --checkpoint-every steps of progress, not
the whole run.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.features import load_backbone
from src.model import build_head
from src.transforms import HELD_IN_TRANSFORMS, TRANSFORM_GRID, apply_transform


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(split_path):
    paths, labels = [], []
    with open(split_path) as f:
        for line in f:
            p, l = line.strip().split("\t")
            paths.append(p)
            labels.append(int(l))
    return paths, labels


class AugmentedImageDataset(Dataset):
    """Loads a raw image, applies one random held-in transform (or none)
    per __getitem__ call, matching Section 3.2's distortion_prob=1.0
    training-time augmentation policy — but generated on the fly here
    instead of pre-cached, since the backbone is no longer frozen."""

    def __init__(self, paths, labels, preprocess, seed):
        self.paths = paths
        self.labels = labels
        self.preprocess = preprocess
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        name = self.rng.choice(HELD_IN_TRANSFORMS)
        severity = self.rng.choice(TRANSFORM_GRID[name])
        img = apply_transform(img, name, severity)
        tensor = self.preprocess(img)
        return tensor, torch.tensor(self.labels[idx], dtype=torch.float32)


def build_lora_backbone(model, rank, alpha, target_blocks):
    n_blocks = len(model.visual.transformer.resblocks)
    target_modules = []
    for i in range(max(0, n_blocks - target_blocks), n_blocks):
        target_modules.append(f"visual.transformer.resblocks.{i}.attn.out_proj")
        target_modules.append(f"visual.transformer.resblocks.{i}.mlp.c_fc")
        target_modules.append(f"visual.transformer.resblocks.{i}.mlp.c_proj")
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules, lora_dropout=0.0)
    peft_model = get_peft_model(model, config)
    return peft_model, target_modules


@torch.no_grad()
def evaluate(peft_model, head, paths, labels, preprocess, device, threshold, batch_size=64):
    peft_model.eval()
    head.eval()
    probs_all = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        emb = peft_model.encode_image(imgs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        logits = head(emb)
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs_all)
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    return acc, auc


def save_checkpoint(path, peft_model, head, opt, epoch, step, best_val_auc, cfg, embedding_dim):
    tmp_path = path + ".tmp.pt"
    torch.save(
        {
            "lora_state_dict": {k: v for k, v in peft_model.state_dict().items() if "lora_" in k},
            "head_state_dict": head.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val_auc": best_val_auc,
            "backbone_family": cfg["backbone"]["family"],
            "backbone_pretrained": cfg["backbone"]["pretrained"],
            "adaptation_method": "lora",
            "embedding_dim": embedding_dim,
            "classification_threshold": cfg["eval"]["classification_threshold"],
            "seed": cfg["seed"],
            "lora_rank": cfg["adaptation"]["lora"]["rank"],
            "lora_target_blocks": cfg["adaptation"]["lora"]["target_blocks"],
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint-out", default="outputs/checkpoints/phase4b_lora.pt")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                         help="Save a resumable checkpoint every N training steps")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override config's train.max_epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Smaller than Phase 4's 256 — every step now runs the real backbone forward+backward")
    parser.add_argument("--lr", type=float, default=1e-4, help="Lower than Phase 4's linear-probe lr, since we're now updating backbone weights")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    backbone, preprocess, embedding_dim = load_backbone(
        cfg["backbone"]["family"], cfg["backbone"]["pretrained"], device
    )
    lora_cfg = cfg["adaptation"]["lora"]
    peft_model, target_modules = build_lora_backbone(
        backbone, lora_cfg["rank"], lora_cfg["rank"] * 2, lora_cfg["target_blocks"]
    )
    peft_model = peft_model.to(device)
    print(f"LoRA targets: {target_modules}")
    peft_model.print_trainable_parameters()

    head = build_head("frozen_linear", embedding_dim).to(device)

    train_paths, train_labels = load_split(os.path.join(cfg["paths"]["splits_dir"], "train.txt"))
    val_paths, val_labels = load_split(os.path.join(cfg["paths"]["splits_dir"], "val.txt"))
    print(f"Train: {len(train_paths)} images, Val: {len(val_paths)} images")

    train_dataset = AugmentedImageDataset(train_paths, train_labels, preprocess, cfg["seed"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    threshold = cfg["eval"]["classification_threshold"]
    max_epochs = args.max_epochs or cfg["train"]["max_epochs"]

    start_epoch, global_step, best_val_auc = 0, 0, -1.0
    if os.path.exists(args.checkpoint_out):
        print(f"Resuming from {args.checkpoint_out}")
        ckpt = torch.load(args.checkpoint_out, map_location=device)
        peft_state = peft_model.state_dict()
        peft_state.update(ckpt["lora_state_dict"])
        peft_model.load_state_dict(peft_state)
        head.load_state_dict(ckpt["head_state_dict"])
        opt.load_state_dict(ckpt["opt_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]
        best_val_auc = ckpt["best_val_auc"]
        print(f"Resumed at epoch {start_epoch}, step {global_step}, best_val_auc so far {best_val_auc:.4f}")

    t_start = time.time()
    for epoch in range(start_epoch, max_epochs):
        peft_model.train()
        head.train()
        epoch_loss = 0.0
        n_seen = 0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            opt.zero_grad()
            emb = peft_model.encode_image(batch_imgs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = head(emb)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch_imgs.size(0)
            n_seen += batch_imgs.size(0)
            global_step += 1

            if global_step % args.checkpoint_every == 0:
                save_checkpoint(args.checkpoint_out, peft_model, head, opt, epoch, global_step, best_val_auc, cfg, embedding_dim)
                elapsed = time.time() - t_start
                print(f"  [checkpoint] epoch {epoch+1} step {global_step} "
                      f"({n_seen}/{len(train_dataset)} images this epoch), elapsed {elapsed:.0f}s")

        val_acc, val_auc = evaluate(peft_model, head, val_paths, val_labels, preprocess, device, threshold)
        print(f"Epoch {epoch+1}/{max_epochs}  train_loss={epoch_loss/n_seen:.4f}  "
              f"val_acc={val_acc:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
        save_checkpoint(args.checkpoint_out, peft_model, head, opt, epoch + 1, global_step, best_val_auc, cfg, embedding_dim)
        print(f"  [checkpoint] end of epoch {epoch+1} saved")

    print(f"\nTraining complete. best_val_auc={best_val_auc:.4f}")


if __name__ == "__main__":
    main()
