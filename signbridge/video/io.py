"""Streaming frame sources (OpenCV video files and image-sequence folders).

Frames are never all loaded at once: ``read_frames`` decodes sequentially and yields
only the requested indices. A frame that cannot be decoded is yielded as ``None`` so
that the preprocessing layer can apply its missing-frame policy.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import cv2
import numpy as np


class VideoUnavailableError(FileNotFoundError):
    """The video file does not exist locally or cannot be opened."""


@dataclass
class VideoInfo:
    path: str
    fps: float | None              # None when the source has no timing (image folders)
    frame_count: int               # container-reported count (may be inaccurate)
    width: int
    height: int
    codec: str | None
    duration: float | None         # frame_count / fps
    decoded_frames: int | None = None  # filled only by a full decode pass
    corrupted_frames: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _fourcc_str(v: float) -> str | None:
    code = int(v)
    if code <= 0:
        return None
    s = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    return s if s.isprintable() else None


def probe_video(path: str | Path, full_decode: bool = False) -> VideoInfo:
    """Read container metadata. With ``full_decode`` every frame is decoded to count
    real/corrupted frames (slow but authoritative)."""
    path = Path(path)
    if not path.exists():
        raise VideoUnavailableError(str(path))
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoUnavailableError(f"OpenCV cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info = VideoInfo(str(path), fps, n, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
                         (n / fps) if fps else None)
        if full_decode:
            ok_n = bad = 0
            while True:
                grabbed = cap.grab()
                if not grabbed:
                    break
                ok, frame = cap.retrieve()
                if ok and frame is not None and frame.size:
                    ok_n += 1
                else:
                    bad += 1
            info.decoded_frames, info.corrupted_frames = ok_n, bad
        return info
    finally:
        cap.release()


class FrameSource(Protocol):
    """Anything that can stream frames by index (video file, image folder, camera buffer)."""

    @property
    def info(self) -> VideoInfo: ...
    @property
    def num_frames(self) -> int: ...
    @property
    def fps(self) -> float | None: ...
    def read_frames(self, indices: Sequence[int] | None = None) -> Iterator[tuple[int, np.ndarray | None]]: ...
    def close(self) -> None: ...


class VideoReader:
    """Lazy OpenCV reader for one video, optionally restricted to [start_time, end_time).

    Indices passed to :meth:`read_frames` are *segment-local* (0 = first frame of the
    segment). Decoding is sequential (``grab`` for skipped frames), which is both faster
    and more accurate than random seeking for H.264.
    """

    def __init__(self, path: str | Path, start_time: float | None = None,
                 end_time: float | None = None, seek: str = "seek"):
        if seek not in ("seek", "grab"):
            raise ValueError("seek must be 'seek' (container seek) or 'grab' (decode from 0)")
        self.seek = seek
        self.path = Path(path)
        self._info = probe_video(self.path)
        fps = self._info.fps
        n = self._info.frame_count
        if (start_time is not None or end_time is not None) and not fps:
            raise ValueError(f"{path}: cannot cut by time, FPS unknown")
        self.start_frame = 0 if start_time is None else max(0, int(math.floor(start_time * fps + 1e-6)))
        self.end_frame = n if end_time is None else min(n, int(math.ceil(end_time * fps - 1e-6)))
        self.requested_start, self.requested_end = start_time, end_time
        # How much of the requested segment lies outside the file (alignment diagnostics)
        self.truncated_tail_s = 0.0 if end_time is None or not fps else max(0.0, end_time - n / fps)
        if self.end_frame <= self.start_frame:
            raise ValueError(f"{path}: empty segment [{start_time}, {end_time}) "
                             f"for a video of {n} frames @ {fps} fps")
        self._cap: cv2.VideoCapture | None = None

    @property
    def info(self) -> VideoInfo:
        return self._info

    @property
    def fps(self) -> float | None:
        return self._info.fps

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    def timestamp(self, local_index: int) -> float | None:
        """Absolute timestamp (seconds in the file) of a segment-local frame index."""
        return None if not self.fps else (self.start_frame + local_index) / self.fps

    def read_frames(self, indices: Sequence[int] | None = None) -> Iterator[tuple[int, np.ndarray | None]]:
        wanted = range(self.num_frames) if indices is None else indices
        wanted = sorted(set(int(i) for i in wanted))
        if wanted and (wanted[0] < 0 or wanted[-1] >= self.num_frames):
            raise IndexError(f"frame index out of range [0, {self.num_frames})")
        cap = cv2.VideoCapture(str(self.path))
        try:
            if self.start_frame and self.seek == "seek":
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            elif self.start_frame:                    # frame-exact: decode-skip from 0
                for _ in range(self.start_frame):
                    if not cap.grab():
                        break
            pos = 0
            for idx in wanted:
                while pos < idx:                      # skip without full decode
                    if not cap.grab():
                        break
                    pos += 1
                ok, frame = cap.read() if pos == idx else (False, None)
                pos += 1
                yield idx, (frame if ok and frame is not None else None)
        finally:
            cap.release()

    def close(self) -> None:  # stateless between calls; kept for the protocol
        pass

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ImageFolderReader:
    """Frame source over an ordered folder of images (e.g. ``data/A/frame_0001.jpg``).

    ``fps`` is ``None`` unless the caller knows it: image folders carry no timing.
    """

    def __init__(self, folder: str | Path, pattern: str = r".*\.(jpg|jpeg|png)$",
                 fps: float | None = None):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise VideoUnavailableError(str(self.folder))
        rx = re.compile(pattern, re.IGNORECASE)
        self.files = sorted(p for p in self.folder.iterdir() if rx.match(p.name))
        if not self.files:
            raise VideoUnavailableError(f"no images in {self.folder}")
        first = cv2.imread(str(self.files[0]))
        h, w = (first.shape[:2] if first is not None else (0, 0))
        n = len(self.files)
        self._info = VideoInfo(str(self.folder), fps, n, w, h, "image-sequence",
                               (n / fps) if fps else None)

    @property
    def info(self) -> VideoInfo:
        return self._info

    @property
    def fps(self) -> float | None:
        return self._info.fps

    @property
    def num_frames(self) -> int:
        return len(self.files)

    def timestamp(self, local_index: int) -> float | None:
        return None if not self.fps else local_index / self.fps

    def read_frames(self, indices: Sequence[int] | None = None) -> Iterator[tuple[int, np.ndarray | None]]:
        wanted = range(self.num_frames) if indices is None else sorted(set(int(i) for i in indices))
        for idx in wanted:
            if idx < 0 or idx >= self.num_frames:
                raise IndexError(f"frame index {idx} out of range [0, {self.num_frames})")
            yield idx, cv2.imread(str(self.files[idx]))  # None if unreadable

    def close(self) -> None:
        pass

    def __enter__(self) -> "ImageFolderReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
