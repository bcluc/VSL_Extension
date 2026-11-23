"""Pose path: per-frame hand keypoints with MediaPipe HandLandmarker.

Uses the MediaPipe ``hand_landmarker.task`` model (``scripts/fetch_mediapipe_models.py``) and
tracks up to two hands, because continuous signing is two-handed. Output per sequence::

    keypoints (T, 2, 21, 3) float32  x, y in [0,1] image coords, z relative depth
    present   (T, 2) bool            False -> that hand was not detected (keypoints = 0)

Hands are ordered by MediaPipe handedness label (index 0 = "Left", 1 = "Right" as
reported by the model); missing hands are never interpolated here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

HANDEDNESS_SLOT = {"Left": 0, "Right": 1}


class HandKeypointExtractor:
    def __init__(self, model_path: str | Path, num_hands: int = 2, min_detection: float = 0.5,
                 min_presence: float = 0.5):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as mp
        self._mp = mp
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=num_hands, min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence, running_mode=mp_vision.RunningMode.IMAGE)
        self._lm = mp_vision.HandLandmarker.create_from_options(opts)

    def extract(self, frames_bgr: Iterable[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
        kps, present = [], []
        for f in frames_bgr:
            k = np.zeros((2, 21, 3), np.float32)
            p = np.zeros(2, bool)
            if f is not None:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
                for lms, handed in zip(res.hand_landmarks, res.handedness):
                    slot = HANDEDNESS_SLOT.get(handed[0].category_name)
                    if slot is None or p[slot]:
                        continue
                    k[slot] = [(lm.x, lm.y, lm.z) for lm in lms]
                    p[slot] = True
            kps.append(k)
            present.append(p)
        if not kps:
            return np.zeros((0, 2, 21, 3), np.float32), np.zeros((0, 2), bool)
        return np.stack(kps), np.stack(present)

    def close(self) -> None:
        self._lm.close()

    def __enter__(self) -> "HandKeypointExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def keypoints_to_model_input(kps: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(T,2,21,3) + (T,2) -> (T, 42, 4): coordinates + per-keypoint presence channel, so a
    pose encoder can distinguish 'hand absent' from 'hand at origin'."""
    pres = np.repeat(present[:, :, None], 21, axis=2)[..., None].astype(np.float32)
    return np.concatenate([kps, pres], -1).reshape(kps.shape[0], 42, 4)
