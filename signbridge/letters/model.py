"""Small MLP over hand-landmark features, and its checkpoint format."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import FEATURE_DIM, FEATURE_VERSION

CHECKPOINT_KIND = "signbridge.letters.mlp"


class LetterMLP(nn.Module):
    def __init__(self, n_classes: int, in_dim: int = FEATURE_DIM, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes))

    def set_normalisation(self, feats: np.ndarray) -> None:
        self.mean.copy_(torch.from_numpy(feats.mean(0)))
        self.std.copy_(torch.from_numpy(feats.std(0) + 1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D) -> (B, C) logits
        return self.net((x - self.mean) / self.std)


def save_letter_model(path: Path, model: LetterMLP, labels: list[str], hidden: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": CHECKPOINT_KIND, "feature_version": FEATURE_VERSION, "labels": labels,
                "hidden": hidden, "state_dict": model.state_dict(), "meta": meta}, path)


def load_letter_model(path: Path) -> tuple[LetterMLP, list[str], dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained letter model at {path}. Record letters with scripts/record_letters.py, "
            "then train with scripts/train_letter_classifier.py.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != CHECKPOINT_KIND or ck.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"{path} is not a {CHECKPOINT_KIND} checkpoint with features {FEATURE_VERSION}")
    model = LetterMLP(len(ck["labels"]), hidden=ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), list(ck["labels"]), ck.get("meta", {})
