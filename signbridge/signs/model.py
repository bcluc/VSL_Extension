"""Small sequence classifier over ``sequence_features`` (CPU-trainable)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SignClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 2 * hidden), nn.GELU(),
                                 nn.Dropout(dropout))
        self.gru = nn.GRU(2 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        h, _ = self.gru(self.inp(x))
        w = torch.softmax(self.attn(h), dim=1)
        return self.head((w * h).sum(dim=1))


def save_checkpoint(path: Path, model: SignClassifier, classes: list[str], t: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "t": t,
                "in_dim": model.inp[1].in_features, "hidden": model.gru.hidden_size, "meta": meta}, path)


def load_checkpoint(path: Path) -> tuple[SignClassifier, list[str], int, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SignClassifier(ck["in_dim"], len(ck["classes"]), hidden=ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["classes"], ck["t"], ck["meta"]


class EnsembleClassifier(nn.Module):
    """Averages the class probabilities of several checkpoints (e.g. the 3 training seeds).

    ``forward`` returns log(mean probability), so ``softmax(forward(x))`` is the mean
    probability and the module is a drop-in replacement for a single ``SignClassifier``.
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("need at least one model")
        self.members = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.stack([torch.softmax(m(x), dim=1) for m in self.members]).mean(dim=0)
        return torch.log(p.clamp_min(1e-12))


def load_ensemble(paths: list[Path]) -> tuple[nn.Module, list[str], int, dict]:
    """One checkpoint -> that model; several -> an ``EnsembleClassifier`` (same classes and T required)."""
    loaded = [load_checkpoint(Path(p)) for p in paths]
    _, classes, t, meta = loaded[0]
    for p, (_, c, tt, _) in zip(paths, loaded):
        if c != classes or tt != t:
            raise ValueError(f"{p}: classes or T differ from {paths[0]}")
    if len(loaded) == 1:
        return loaded[0][0], classes, t, meta
    model = EnsembleClassifier([m for m, *_ in loaded]).eval()
    return model, classes, t, {**meta, "ensemble": [Path(p).name for p in paths]}
