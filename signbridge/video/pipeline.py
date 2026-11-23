"""Configurable, dataset-agnostic video preprocessing.

FrameSource --(sample indices)--> decode (streaming) --> missing-frame policy
           --> resize + normalize --> ProcessedClip (T, C, H, W) + valid mask + timestamps
           --> temporal windows

Two entry points:
* :meth:`VideoPreprocessor.process`        whole (short) clip -> one array, capped by ``max_frames``
* :meth:`VideoPreprocessor.stream_windows` long videos -> windows yielded one at a time
  (memory bounded by one window, never the whole video)
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .io import FrameSource
from .sampling import make_windows, sample_indices
from .transforms import IMAGENET_MEAN, IMAGENET_STD, to_model_input

MISSING_POLICIES = ("repeat_last", "zero", "drop")


class CorruptVideoError(RuntimeError):
    """Too many frames could not be decoded."""


@dataclass
class PreprocessConfig:
    sampling: str = "fps"                 # all | uniform | stride | fps
    target_fps: float | None = 12.0       # for sampling="fps"
    num_samples: int | None = None        # for sampling="uniform"
    stride: int | None = None             # for sampling="stride"
    max_frames: int | None = 256          # memory guard for process()
    image_size: tuple[int, int] = (224, 224)  # (H, W)
    keep_aspect: bool = True
    color: str = "rgb"
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD
    window_size: int = 16
    window_stride: int = 8
    missing_frame_policy: str = "repeat_last"
    max_missing_ratio: float = 0.2
    fallback_fps: float | None = None     # used only when a source has no FPS (image folders)

    def __post_init__(self) -> None:
        self.image_size = tuple(self.image_size)  # JSON gives lists
        self.mean, self.std = tuple(self.mean), tuple(self.std)
        if self.missing_frame_policy not in MISSING_POLICIES:
            raise ValueError(f"missing_frame_policy must be one of {MISSING_POLICIES}")

    @classmethod
    def from_json(cls, path: str | Path) -> "PreprocessConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessedClip:
    frames: np.ndarray                 # (T, C, H, W) float32
    valid_mask: np.ndarray             # (T,) bool, False = frame missing/undecodable
    frame_indices: np.ndarray          # (T,) segment-local source indices
    timestamps: np.ndarray | None      # (T,) seconds in the source file, None if no FPS
    effective_fps: float | None        # temporal rate of the processed sequence
    windows: list[tuple[int, int]] = field(default_factory=list)
    n_missing: int = 0

    @property
    def length(self) -> int:
        return int(self.frames.shape[0])

    def window_array(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (frames, mask) for window i, zero-padded past the end."""
        s, e = self.windows[i]
        w = e - s
        out = np.zeros((w,) + self.frames.shape[1:], dtype=self.frames.dtype)
        mask = np.zeros(w, dtype=bool)
        real = self.frames[s:min(e, self.length)]
        out[: len(real)] = real
        mask[: len(real)] = self.valid_mask[s:min(e, self.length)]
        return out, mask


@dataclass
class Window:
    index: int
    start: int                          # position in the sampled sequence
    end: int
    frames: np.ndarray                  # (window_size, C, H, W), zero padded
    mask: np.ndarray                    # (window_size,) bool
    timestamps: np.ndarray | None


class VideoPreprocessor:
    def __init__(self, config: PreprocessConfig | None = None):
        self.cfg = config or PreprocessConfig()

    # ------------------------------------------------------------------ helpers
    def _indices(self, source: FrameSource) -> tuple[np.ndarray, float | None]:
        c = self.cfg
        src_fps = source.fps or c.fallback_fps
        idx = sample_indices(source.num_frames, c.sampling, num_samples=c.num_samples,
                             stride=c.stride, src_fps=src_fps, target_fps=c.target_fps,
                             max_frames=None)
        eff = None
        if src_fps:
            eff = {"all": src_fps, "stride": src_fps / (c.stride or 1), "fps": c.target_fps,
                   "uniform": src_fps * len(idx) / max(source.num_frames, 1)}[c.sampling]
        return idx, eff

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        c = self.cfg
        return to_model_input(frame, c.image_size, c.keep_aspect, c.mean, c.std, c.color)

    def _empty_frame(self) -> np.ndarray:
        ch = 1 if self.cfg.color == "gray" else 3
        return np.zeros((ch,) + tuple(self.cfg.image_size), dtype=np.float32)

    def _timestamps(self, source: FrameSource, idx: np.ndarray) -> np.ndarray | None:
        ts = getattr(source, "timestamp", None)
        if ts is None or not source.fps:
            return None
        return np.array([ts(int(i)) for i in idx], dtype=np.float64)

    # ------------------------------------------------------------------ whole clip
    def process(self, source: FrameSource) -> ProcessedClip:
        c = self.cfg
        idx, eff = self._indices(source)
        if c.max_frames and len(idx) > c.max_frames:
            keep = np.floor((np.arange(c.max_frames) + 0.5) * len(idx) / c.max_frames).astype(np.int64)
            if eff:
                eff = eff * c.max_frames / len(idx)
            idx = idx[keep]
        # duplicates (uniform sampling of short clips) are decoded once
        decoded: dict[int, np.ndarray | None] = {i: f for i, f in source.read_frames(np.unique(idx))}
        frames, mask = [], []
        for i in idx:
            f = decoded.get(int(i))
            mask.append(f is not None)
            frames.append(self._transform(f) if f is not None else None)
        n_missing = int(len(mask) - sum(mask))
        self._check_missing(n_missing, len(mask), source)
        frames, mask, idx = self._apply_missing_policy(frames, np.array(mask, bool), idx)
        clip = ProcessedClip(
            frames=np.stack(frames) if frames else np.zeros((0,) + self._empty_frame().shape, np.float32),
            valid_mask=mask, frame_indices=idx, timestamps=self._timestamps(source, idx),
            effective_fps=eff, n_missing=n_missing)
        clip.windows = make_windows(clip.length, c.window_size, c.window_stride)
        return clip

    def _check_missing(self, n_missing: int, total: int, source: FrameSource) -> None:
        if total and n_missing / total > self.cfg.max_missing_ratio:
            raise CorruptVideoError(f"{source.info.path}: {n_missing}/{total} sampled frames "
                                    f"undecodable (> {self.cfg.max_missing_ratio:.0%})")

    def _apply_missing_policy(self, frames: list, mask: np.ndarray, idx: np.ndarray):
        policy = self.cfg.missing_frame_policy
        if mask.all():
            return frames, mask, idx
        if policy == "drop":
            keep = np.flatnonzero(mask)
            return [frames[k] for k in keep], mask[keep], idx[keep]
        if policy == "zero" or not mask.any():
            return [f if f is not None else self._empty_frame() for f in frames], mask, idx
        # repeat_last (back-fill leading gaps with the first valid frame)
        first_valid = frames[int(np.argmax(mask))]
        out, last = [], first_valid
        for f in frames:
            last = f if f is not None else last
            out.append(last)
        return out, mask, idx

    # ------------------------------------------------------------------ streaming
    def stream_windows(self, source: FrameSource) -> Iterator[Window]:
        """Yield fixed-size windows while decoding sequentially.

        Peak memory holds at most ``window_size`` processed frames. With policy
        ``drop`` missing frames are zero-filled and masked (window geometry must stay fixed).
        """
        c = self.cfg
        idx, _ = self._indices(source)
        windows = make_windows(len(idx), c.window_size, c.window_stride)
        if not windows:
            return
        ts_all = self._timestamps(source, idx)
        buf: OrderedDict[int, tuple[np.ndarray | None, bool]] = OrderedDict()  # pos -> (frame, valid)
        last_valid: np.ndarray | None = None
        n_missing, w_i = 0, 0
        positions_by_src: dict[int, list[int]] = {}
        for pos, src in enumerate(idx):
            positions_by_src.setdefault(int(src), []).append(pos)

        def emit(wi: int) -> Window:
            s, e = windows[wi]
            arr = np.zeros((c.window_size,) + self._empty_frame().shape, np.float32)
            m = np.zeros(c.window_size, bool)
            for p in range(s, min(e, len(idx))):
                f, valid = buf.get(p, (None, False))
                if f is not None:
                    arr[p - s] = f
                m[p - s] = valid
            ts = None if ts_all is None else ts_all[s:min(e, len(idx))]
            return Window(wi, s, e, arr, m, ts)

        for src_i, frame in source.read_frames(np.unique(idx)):
            f = self._transform(frame) if frame is not None else None
            if f is None:
                n_missing += len(positions_by_src[src_i])
                f_fill = last_valid if self.cfg.missing_frame_policy == "repeat_last" else None
            else:
                last_valid = f_fill = f
            for pos in positions_by_src[src_i]:
                buf[pos] = (f_fill, f is not None)
                # emit all windows whose last real position is now available
                while w_i < len(windows) and pos >= min(windows[w_i][1], len(idx)) - 1:
                    yield emit(w_i)
                    w_i += 1
                    nxt = windows[w_i][0] if w_i < len(windows) else len(idx)
                    for old in [k for k in buf if k < nxt]:
                        del buf[old]
        self._check_missing(n_missing, len(idx), source)
        while w_i < len(windows):  # source ended early (fewer decodable frames than reported)
            yield emit(w_i)
            w_i += 1
