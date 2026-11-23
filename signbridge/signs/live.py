"""Sign-by-sign live recognition state machine (camera-independent).

Feed frames (landmarks) with timestamps; the session emits events:

* ``SignEvent``     a segment ended and was classified (accepted or rejected by confidence);
* ``SentenceEvent`` a pause after one or more accepted signs; the phrase table composes text.

Segmentation rule (the user signs one sign at a time and lowers the hands in between):
a frame is *active* when a hand is visible and its wrist is above the rest line
(hip level estimated from the shoulders). A segment starts after ``start_frames`` active
frames and ends after ``end_gap_s`` seconds without activity.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .features import RawSequence, sequence_features
from .phrases import Composed, compose


@dataclass
class SignEvent:
    t: float
    gloss: str
    confidence: float
    accepted: bool
    top3: list[tuple[str, float]]
    n_frames: int
    duration_s: float
    ended_by: str = "rest"  # "rest" (hands lowered), "max" (max_sign_s reached), "flush"


@dataclass
class SentenceEvent:
    t: float
    glosses: list[str]
    composed: Composed


@dataclass
class LiveConfig:
    start_frames: int = 2
    end_gap_s: float = 0.4
    min_sign_s: float = 0.25
    max_sign_s: float = 4.0
    sentence_gap_s: float = 2.0
    min_confidence: float = 0.45
    preroll: int = 3
    # rest line = shoulder_y + rest_line * shoulder_width. 1.1: the highest wrist of 99.6 % of
    # train+val sign cores is above it, while hands resting on a desk (about 1.3 for the demo
    # user) are below it. Was 1.6 (hip level), which kept desk-resting hands 'active'.
    rest_line: float = 1.1
    # debounce: a frame counts as active when >= active_min of the last active_window raw frames
    # are active (1/1 = no debounce). Smooths hand-detection flicker at the frame edge.
    active_window: int = 1
    active_min: int = 1


def rest_line_y(pose: np.ndarray, pose_ok: bool, cfg: LiveConfig) -> float | None:
    """Image y (0 = top, 1 = bottom) below which a wrist counts as resting; None without a pose."""
    if not pose_ok:
        return None
    sh_y = (pose[11, 1] + pose[12, 1]) / 2
    sh_w = abs(pose[11, 0] - pose[12, 0])
    return float(sh_y + cfg.rest_line * max(sh_w, 0.05))


def is_active(pose: np.ndarray, hands: np.ndarray, present: np.ndarray, pose_ok: bool, cfg: LiveConfig) -> bool:
    if not present.any():
        return False
    line = rest_line_y(pose, pose_ok, cfg)
    if line is None:
        return True
    wrists_y = hands[present][:, 0, 1]
    return bool((wrists_y < line).any())


@dataclass
class SignSession:
    model: torch.nn.Module
    classes: list[str]
    t: int = 32
    cfg: LiveConfig = field(default_factory=LiveConfig)

    def __post_init__(self):
        self._pre: deque = deque(maxlen=self.cfg.preroll)
        self._buf: list = []
        self._times: list[float] = []
        self._active_run = 0
        self._last_active = None
        self._in_sign = False
        self._glosses: list[str] = []
        self._last_sign_end = None
        self._raw: deque = deque(maxlen=max(1, self.cfg.active_window))

    @property
    def in_sign(self) -> bool:
        return self._in_sign

    @property
    def pending_glosses(self) -> list[str]:
        return list(self._glosses)

    def classify(self, seq: RawSequence) -> tuple[np.ndarray, list[int]]:
        x = torch.from_numpy(sequence_features(seq, self.t)[None])
        with torch.no_grad():
            p = torch.softmax(self.model(x), 1)[0].numpy()
        return p, list(np.argsort(-p)[:3])

    def push(self, now: float, pose, hands, present, pose_ok) -> list:
        events: list = []
        frame = (pose, hands, present, pose_ok)
        self._raw.append(is_active(pose, hands, present, pose_ok, self.cfg))
        act = sum(self._raw) >= self.cfg.active_min
        self._active_run = self._active_run + 1 if act else 0
        if act:
            self._last_active = now
        if not self._in_sign:
            self._pre.append((now, frame))
            if self._active_run >= self.cfg.start_frames:
                self._in_sign = True
                self._times = [t for t, _ in self._pre]
                self._buf = [f for _, f in self._pre]
            elif (self._glosses and self._last_sign_end is not None
                  and now - self._last_sign_end >= self.cfg.sentence_gap_s):
                events.append(SentenceEvent(now, list(self._glosses), compose(self._glosses)))
                self._glosses = []
            return events
        self._buf.append(frame)
        self._times.append(now)
        dur = now - self._times[0]
        ended = (self._last_active is not None and now - self._last_active >= self.cfg.end_gap_s)
        if ended or dur >= self.cfg.max_sign_s:
            events.append(self._finish(now, "rest" if ended else "max"))
        return events

    def _finish(self, now: float, ended_by: str = "rest") -> SignEvent:
        seq = RawSequence.stack(self._buf)
        dur = self._times[-1] - self._times[0]
        self._in_sign = False
        self._pre.clear()
        self._active_run = 0
        self._last_sign_end = now
        if dur < self.cfg.min_sign_s or not seq.hand_present.any():
            return SignEvent(now, "", 0.0, False, [], len(seq), dur, ended_by)
        p, top = self.classify(seq)
        g, c = self.classes[top[0]], float(p[top[0]])
        ok = c >= self.cfg.min_confidence
        if ok:
            self._glosses.append(g)
        return SignEvent(now, g, c, ok, [(self.classes[i], float(p[i])) for i in top], len(seq), dur, ended_by)

    def flush(self, now: float) -> list:
        """Force the end of the current sign and sentence (e.g. key press or end of video)."""
        events: list = []
        if self._in_sign:
            events.append(self._finish(now, "flush"))
        if self._glosses:
            events.append(SentenceEvent(now, list(self._glosses), compose(self._glosses)))
            self._glosses = []
        return events
