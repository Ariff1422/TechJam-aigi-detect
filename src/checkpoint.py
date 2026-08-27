"""Shared checkpoint load helper — used by infer.py and robustness_eval.py
so both apply the exact same backbone+head reconstruction logic."""
import torch

from src.features import load_backbone
from src.model import build_head


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a checkpoint saved by toy_train.py / train.py and reconstruct
    the (backbone, preprocess, head) needed for inference.

    Returns (model, preprocess, head, embedding_dim).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model, preprocess, embedding_dim = load_backbone(
        ckpt["backbone_family"], ckpt["backbone_pretrained"], device
    )
    assert embedding_dim == ckpt["embedding_dim"], (
        f"Embedding dim mismatch: backbone produces {embedding_dim}, "
        f"checkpoint expects {ckpt['embedding_dim']}"
    )

    head = build_head(ckpt["adaptation_method"], embedding_dim)
    head.load_state_dict(ckpt["head_state_dict"])
    head = head.to(device).eval()

    return model, preprocess, head, embedding_dim
