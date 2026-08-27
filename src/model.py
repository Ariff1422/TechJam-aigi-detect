"""Classifier heads on frozen backbone embeddings. PLAN.md Section 1.2."""
import torch.nn as nn


class LinearHead(nn.Module):
    """MVP head: single linear layer, sigmoid output. PLAN.md Section 1.2."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits; apply sigmoid outside for probs


class MLPHead(nn.Module):
    """Stretch upgrade head: embedding_dim -> 256 -> 1, ReLU, dropout 0.3.

    PLAN.md Section 1.2 — not used in Phase 1, included for config completeness.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_head(method: str, embedding_dim: int, **kwargs):
    if method == "frozen_linear":
        return LinearHead(embedding_dim)
    if method == "frozen_mlp":
        return MLPHead(embedding_dim, kwargs.get("mlp_hidden_dim", 256), kwargs.get("dropout", 0.3))
    raise ValueError(f"Unsupported adaptation method for Phase 1: {method}")
