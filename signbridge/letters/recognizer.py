"""Frame -> hand landmarks -> letter probabilities, plus the "hold to type" logic."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .features import letter_features
from .labels import append_letter, display
from .model import load_letter_model


@dataclass
class HandObservation:
    landmarks: np.ndarray   # (21, 3) normalised image coordinates of the mirrored view
    is_left: bool           # MediaPipe handedness on the mirrored view
    aspect: float           # image width / height
    score: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xy = self.landmarks[:, :2]
        (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
        return float(x0), float(y0), float(x1), float(y1)


class HandDetector:
    """One hand per frame with MediaPipe HandLandmarker (IMAGE mode). Feed the mirrored view."""

    def __init__(self, model_path: str | Path, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision as mpv
        if not Path(model_path).exists():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path} "
                                    "(run scripts/fetch_mediapipe_models.py)")
        self._mp = mp
        self._lm = mpv.HandLandmarker.create_from_options(mpv.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mpv.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf))

    def detect(self, bgr: np.ndarray) -> HandObservation | None:
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect(img)
        if not res.hand_landmarks:
            return None
        lms = np.array([(p.x, p.y, p.z) for p in res.hand_landmarks[0]], np.float32)
        cat = res.handedness[0][0]
        return HandObservation(lms, cat.category_name == "Left", w / h, float(cat.score))

    def close(self) -> None:
        self._lm.close()


@dataclass
class LetterPrediction:
    hand: HandObservation | None
    probs: np.ndarray | None = None          # (C,)
    labels: list[str] = field(default_factory=list)
    ms: float = 0.0

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        if self.probs is None:
            return []
        idx = np.argsort(-self.probs)[:k]
        return [(self.labels[i], float(self.probs[i])) for i in idx]


class LetterRecognizer:
    def __init__(self, checkpoint: str | Path, hand_model: str | Path):
        self.model, self.labels, self.meta = load_letter_model(Path(checkpoint))
        self.detector = HandDetector(hand_model)

    @torch.no_grad()
    def predict(self, bgr_mirrored: np.ndarray) -> LetterPrediction:
        t0 = time.perf_counter()
        hand = self.detector.detect(bgr_mirrored)
        if hand is None:
            return LetterPrediction(None, labels=self.labels, ms=(time.perf_counter() - t0) * 1000)
        x = torch.from_numpy(letter_features(hand.landmarks, hand.is_left, hand.aspect))
        probs = torch.softmax(self.model(x), -1)[0].numpy()
        return LetterPrediction(hand, probs, self.labels, (time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        self.detector.close()


class LetterTyper:
    """Types a letter once its averaged probability stays above ``threshold`` for ``hold_s``.

    The same letter is typed again only after the hand leaves the view or another letter
    has been typed.
    """

    def __init__(self, labels: list[str], window: int = 6, threshold: float = 0.6, hold_s: float = 0.8):
        self.labels, self.threshold, self.hold_s = labels, threshold, hold_s
        self.buf: deque[np.ndarray] = deque(maxlen=window)
        self.text = ""
        self._cand: str | None = None
        self._since = 0.0
        self._typed: str | None = None

    def current(self) -> tuple[str | None, float]:
        if not self.buf:
            return None, 0.0
        mean = np.mean(self.buf, axis=0)
        i = int(np.argmax(mean))
        return (self.labels[i], float(mean[i])) if mean[i] >= self.threshold else (None, float(mean[i]))

    def push(self, pred: LetterPrediction, now: float) -> str | None:
        """Feed one prediction; returns the label id when a letter is typed."""
        if pred.probs is None:
            self.buf.clear()
            self._cand = self._typed = None
            return None
        self.buf.append(pred.probs)
        lab, _ = self.current()
        if lab != self._cand:
            self._cand, self._since = lab, now
            return None
        if lab is not None and lab != self._typed and now - self._since >= self.hold_s:
            self._typed = lab
            self.text = append_letter(self.text, lab)
            return lab
        return None

    def space(self) -> None:
        self.text += " "

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""


__all__ = ["HandDetector", "HandObservation", "LetterPrediction", "LetterRecognizer", "LetterTyper", "display"]
