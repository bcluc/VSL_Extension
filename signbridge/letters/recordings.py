"""On-disk format of recorded letter takes (landmarks only, never images).

    <root>/<signer>/<LABEL_ID>/take_<YYYYmmdd-HHMMSS-ffffff>.npz
        landmarks (T, 21, 3) float32   frames where a hand was detected
        is_left   (T,) bool
        aspect    float                image width / height
        fps       float                capture rate during the take
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Take:
    path: Path
    signer: str
    label: str
    landmarks: np.ndarray
    is_left: np.ndarray
    aspect: float

    def __len__(self) -> int:
        return len(self.landmarks)


def save_take(root: Path, signer: str, label: str, landmarks: list[np.ndarray], is_left: list[bool],
              aspect: float, fps: float) -> Path:
    d = Path(root) / signer / label
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"take_{dt.datetime.now():%Y%m%d-%H%M%S-%f}.npz"
    np.savez_compressed(path, landmarks=np.stack(landmarks).astype(np.float32),
                        is_left=np.array(is_left, bool), aspect=np.float32(aspect), fps=np.float32(fps))
    return path


def load_takes(root: Path, labels: list[str] | None = None) -> list[Take]:
    takes = []
    for path in sorted(Path(root).glob("*/*/take_*.npz")):
        label, signer = path.parent.name, path.parent.parent.name
        if labels is not None and label not in labels:
            continue
        z = np.load(path)
        if len(z["landmarks"]) == 0:
            continue
        takes.append(Take(path, signer, label, z["landmarks"], z["is_left"], float(z["aspect"])))
    return takes


def count_takes(root: Path, signer: str) -> dict[str, int]:
    base = Path(root) / signer
    if not base.is_dir():
        return {}
    return {d.name: len(list(d.glob("take_*.npz"))) for d in base.iterdir() if d.is_dir()}
