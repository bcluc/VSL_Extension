"""Video-to-annotation validation checks (dataset-agnostic)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..video.io import VideoUnavailableError, probe_video
from .base import SignDataset


@dataclass
class SampleCheck:
    sample_id: str
    ok: bool
    video_available: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_sample(ds: SignDataset, sample_id: str, full_decode: bool = False,
                    duration_tolerance_s: float = 0.25) -> SampleCheck:
    """Checks: video present + decodable, FPS/frames sane, segment inside the video,
    annotation present for the declared status, clip duration consistent with the
    annotation interval when the dataset records one (``meta.original_start/end``)."""
    rec = ds.get_sample(sample_id).record
    chk = SampleCheck(sample_id, ok=True, video_available=Path(rec.video_path).exists())
    ann = ds.load_annotations(sample_id)
    chk.info["annotation_status"] = ann.status.value
    if ann.status.value in ("translation_only", "gloss_and_translation") and not ann.target_text:
        chk.errors.append("status says translation but target_text empty")
    if ann.status.value in ("gloss_only", "gloss_and_translation", "isolated_label") and not ann.gloss_sequence:
        chk.errors.append("status says gloss but gloss_sequence empty")
    if not chk.video_available:
        chk.errors.append("video not available locally")
        chk.ok = False
        return chk
    p = Path(rec.video_path)
    if p.is_dir():  # image sequence
        src = ds.load_video(sample_id)
        unreadable = sum(1 for _, f in src.read_frames() if f is None)
        chk.info.update(frames=src.num_frames, unreadable_frames=unreadable, fps=None)
        if unreadable:
            chk.errors.append(f"{unreadable} unreadable images")
        chk.ok = not chk.errors
        return chk
    try:
        vi = probe_video(p, full_decode=full_decode)
    except VideoUnavailableError as e:
        chk.errors.append(str(e))
        chk.ok = False
        return chk
    chk.info.update(fps=vi.fps, frame_count=vi.frame_count, duration=vi.duration,
                    width=vi.width, height=vi.height, codec=vi.codec,
                    decoded_frames=vi.decoded_frames, corrupted_frames=vi.corrupted_frames)
    if not vi.fps or vi.frame_count <= 0:
        chk.errors.append("invalid fps/frame_count")
    if full_decode and vi.decoded_frames is not None:
        if vi.corrupted_frames:
            chk.errors.append(f"{vi.corrupted_frames} corrupted frames")
        if vi.decoded_frames != vi.frame_count:
            chk.warnings.append(f"decoded {vi.decoded_frames} != reported {vi.frame_count}")
    if rec.fps and vi.fps and abs(rec.fps - vi.fps) > 1e-3:
        chk.errors.append(f"manifest fps {rec.fps} != probed {vi.fps}")
    # segment inside video
    if rec.end_time is not None and vi.duration is not None:
        overflow = rec.end_time - vi.duration
        chk.info["segment_overflow_s"] = round(overflow, 3)
        if overflow > duration_tolerance_s:
            chk.errors.append(f"segment end {rec.end_time:.2f}s beyond video end {vi.duration:.2f}s")
    # whole-file clip vs annotated interval (How2Sign clips cut with original timestamps)
    if rec.start_time is None and rec.end_time is None and vi.duration is not None:
        os_, oe = rec.meta.get("original_start"), rec.meta.get("original_end")
        if os_ is not None and oe is not None:
            diff = vi.duration - (oe - os_)
            chk.info["clip_minus_annotation_s"] = round(diff, 3)
            if abs(diff) > duration_tolerance_s:
                chk.warnings.append(f"clip duration differs from annotation interval by {diff:+.2f}s")
    chk.ok = not chk.errors
    return chk


def _small_gray(frame: np.ndarray, size=(64, 36)) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)


def locate_clip_in_video(clip_path: str | Path, video_path: str | Path, expected_start: float,
                         search_s: float = 3.0, n_probe: int = 8) -> dict[str, Any]:
    """Find where a sentence clip's first frames occur in its full video.

    Returns the best-matching start time and the mean absolute pixel difference (MAD,
    0-255 scale, on 64x36 grayscale) at the best offset and at the expected offset.
    This measures real alignment between released clips and annotation timestamps.
    """
    clip = cv2.VideoCapture(str(clip_path))
    probe = []
    while len(probe) < n_probe:
        ok, f = clip.read()
        if not ok:
            break
        probe.append(_small_gray(f))
    clip.release()
    if not probe:
        raise VideoUnavailableError(f"cannot decode {clip_path}")
    vi = probe_video(video_path)
    fps = vi.fps
    lo = max(0, int(round((expected_start - search_s) * fps)))
    hi = min(vi.frame_count - len(probe), int(round((expected_start + search_s) * fps)))
    cap = cv2.VideoCapture(str(video_path))
    for _ in range(lo):  # frame-exact positioning
        cap.grab()
    frames = []
    for _ in range(hi - lo + len(probe)):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(_small_gray(f))
    cap.release()
    probe_arr = np.stack(probe)
    scores = []
    for off in range(0, len(frames) - len(probe) + 1):
        scores.append(float(np.mean(np.abs(np.stack(frames[off:off + len(probe)]) - probe_arr))))
    if not scores:
        return {"best_start_s": None, "error": "search window empty"}
    best = int(np.argmin(scores))
    exp_off = int(round(expected_start * fps)) - lo
    return {
        "expected_start_s": expected_start,
        "best_start_s": round((lo + best) / fps, 4),
        "offset_s": round((lo + best) / fps - expected_start, 4),
        "offset_frames": lo + best - int(round(expected_start * fps)),
        "mad_best": round(scores[best], 3),
        "mad_at_expected": round(scores[exp_off], 3) if 0 <= exp_off < len(scores) else None,
        "mad_median_in_window": round(float(np.median(scores)), 3),
        "fps": fps,
    }
