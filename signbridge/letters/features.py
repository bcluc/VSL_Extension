"""Hand landmarks -> fixed feature vector for static letter classification.

Input per frame: MediaPipe HandLandmarker output on the **mirrored** (selfie) view, i.e.
21 points (x, y in [0, 1] image coordinates, z relative depth), whether MediaPipe called the
hand "Left", and the image aspect ratio width / height.

Features (``FEATURE_DIM`` = 83), all invariant to where the hand is and how big it appears:

* 63: the 21 points relative to the wrist, x scaled by the aspect ratio, divided by the
  wrist -> middle-finger-knuckle length.
* 10: distances between the five fingertips (same unit).
* 10: bend of each finger at its two middle joints (cosine between consecutive bones).

The hand's rotation is kept on purpose: some letters differ mainly in orientation.

Handedness: MediaPipe's left/right label is unreliable for some handshapes (it flips between
takes and even between frames of one take), so v2 features do **not** mirror by that label.
Training adds random horizontal mirroring instead (``augment``), so the model learns either
hand. ``mirror_left=True`` restores the label-based mirroring of v1.
"""
from __future__ import annotations

import itertools

import numpy as np

WRIST, MIDDLE_MCP = 0, 9
TIPS = [4, 8, 12, 16, 20]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
_TIP_PAIRS = list(itertools.combinations(TIPS, 2))
FEATURE_DIM = 63 + len(_TIP_PAIRS) + 2 * len(FINGERS)
FEATURE_VERSION = "letters_v2"


def normalise(lms: np.ndarray, is_left: np.ndarray | bool, aspect: np.ndarray | float,
              mirror_left: bool = False) -> np.ndarray:
    """(N, 21, 3) landmarks -> wrist-centred, size-normalised points."""
    p = np.asarray(lms, np.float32).copy()
    if p.ndim == 2:
        p = p[None]
    n = len(p)
    p[..., 0] *= np.broadcast_to(np.asarray(aspect, np.float32), (n,))[:, None]
    p -= p[:, WRIST:WRIST + 1]
    if mirror_left:
        left = np.broadcast_to(np.asarray(is_left, bool), (n,))
        p[left, :, 0] *= -1.0
    size = np.linalg.norm(p[:, MIDDLE_MCP, :2], axis=-1)
    return p / np.maximum(size, 1e-4)[:, None, None]


def letter_features(lms: np.ndarray, is_left: np.ndarray | bool, aspect: np.ndarray | float,
                    mirror_left: bool = False) -> np.ndarray:
    """(N, 21, 3) or (21, 3) landmarks -> (N, FEATURE_DIM) float32."""
    p = normalise(lms, is_left, aspect, mirror_left)
    n = len(p)
    tips = np.stack([np.linalg.norm(p[:, a] - p[:, b], axis=-1) for a, b in _TIP_PAIRS], axis=1)
    bends = []
    for f in FINGERS:
        bones = np.diff(p[:, f], axis=1)                                    # (N, 3, 3)
        bones /= np.maximum(np.linalg.norm(bones, axis=-1, keepdims=True), 1e-6)
        bends.append((bones[:, :-1] * bones[:, 1:]).sum(-1))                # (N, 2)
    feats = np.concatenate([p.reshape(n, -1), tips, np.concatenate(bends, axis=1)], axis=1)
    return feats.astype(np.float32)


def augment(lms: np.ndarray, rng: np.random.Generator, rot_deg: float = 12.0, jitter: float = 0.004,
            scale: float = 0.1, mirror_p: float = 0.5) -> np.ndarray:
    """Random horizontal mirror, in-plane rotation about the wrist, per-axis scale and point
    jitter (training only)."""
    p = np.asarray(lms, np.float32).copy()
    n = len(p)
    flip = rng.random(n) < mirror_p
    p[flip, :, 0] = 2 * p[flip, WRIST:WRIST + 1, 0] - p[flip, :, 0]
    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg, n))
    c, s = np.cos(a), np.sin(a)
    w = p[:, WRIST:WRIST + 1, :2]
    xy = p[..., :2] - w
    rot = np.stack([xy[..., 0] * c[:, None] - xy[..., 1] * s[:, None],
                    xy[..., 0] * s[:, None] + xy[..., 1] * c[:, None]], axis=-1)
    rot *= rng.uniform(1 - scale, 1 + scale, (n, 1, 2))
    p[..., :2] = rot + w
    p += rng.normal(0, jitter, p.shape).astype(np.float32)
    return p
