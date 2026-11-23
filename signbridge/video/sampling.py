"""Frame-index sampling and temporal window generation (pure functions, no I/O)."""
from __future__ import annotations

import math

import numpy as np

STRATEGIES = ("all", "uniform", "stride", "fps")


def sample_indices(num_frames: int, strategy: str = "all", *, num_samples: int | None = None,
                   stride: int | None = None, src_fps: float | None = None,
                   target_fps: float | None = None, max_frames: int | None = None) -> np.ndarray:
    """Return sorted segment-local frame indices to decode.

    all      every frame
    uniform  exactly ``num_samples`` indices spread over the clip (repeats when the clip
             is shorter than ``num_samples``; used for fixed-length clip models)
    stride   every ``stride``-th frame
    fps      resample from ``src_fps`` to ``target_fps`` (nearest frame). Makes sequences
             from 24 fps and 30 fps sources comparable in time.
    ``max_frames`` then caps the length by uniform subsampling (memory guard).
    """
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if strategy == "all":
        idx = np.arange(num_frames)
    elif strategy == "uniform":
        if not num_samples or num_samples <= 0:
            raise ValueError("uniform sampling needs num_samples > 0")
        idx = np.floor((np.arange(num_samples) + 0.5) * num_frames / num_samples).astype(np.int64)
    elif strategy == "stride":
        if not stride or stride <= 0:
            raise ValueError("stride sampling needs stride > 0")
        idx = np.arange(0, num_frames, stride)
    elif strategy == "fps":
        if not src_fps or not target_fps:
            raise ValueError("fps sampling needs src_fps and target_fps (source FPS unknown?)")
        n_out = max(1, int(math.floor(num_frames * target_fps / src_fps + 1e-6)))
        idx = np.round(np.arange(n_out) * src_fps / target_fps).astype(np.int64)
    else:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    idx = np.clip(idx, 0, num_frames - 1)
    if max_frames and len(idx) > max_frames:
        keep = np.floor((np.arange(max_frames) + 0.5) * len(idx) / max_frames).astype(np.int64)
        idx = idx[keep]
    return idx.astype(np.int64)


def make_windows(length: int, window: int, stride: int, *, pad_last: bool = True,
                 min_last: int = 1) -> list[tuple[int, int]]:
    """Sliding windows ``[start, end)`` over a sequence of ``length`` steps.

    With ``pad_last`` the final window may extend past ``length`` (caller pads + masks);
    otherwise a final partial window shorter than ``min_last`` is dropped. A sequence
    shorter than one window yields a single (padded) window.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be > 0")
    if length <= 0:
        return []
    starts = list(range(0, max(length - window, 0) + 1, stride))
    last_end = starts[-1] + window
    if last_end < length:  # tail not covered yet
        tail_start = starts[-1] + stride
        if pad_last or length - tail_start >= min_last:
            starts.append(tail_start)
    windows = [(s, s + window) for s in starts]
    if not pad_last:
        windows = [(s, min(e, length)) for s, e in windows]
    return windows


def time_to_frame_range(start_time: float | None, end_time: float | None, fps: float,
                        num_frames: int) -> tuple[int, int]:
    """Map a [start, end) second interval to a clamped [start_frame, end_frame) range."""
    s = 0 if start_time is None else max(0, int(math.floor(start_time * fps + 1e-6)))
    e = num_frames if end_time is None else min(num_frames, int(math.ceil(end_time * fps - 1e-6)))
    return s, max(s, e)


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0
