"""Shared fixtures: synthetic videos / dataset trees (no real data needed) and the
optional real How2Sign subset (tests marked ``realdata`` skip when absent)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

H2S_ROOT = Path(os.environ.get("SIGNBRIDGE_H2S_ROOT", REPO.parent / "datasets" / "How2Sign"))


def write_video(path: Path, n_frames: int, fps: float, size=(64, 48)) -> Path:
    """Frame i has uniform gray level (i * 5) % 250 -> decoded content identifies the index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert vw.isOpened(), "OpenCV VideoWriter (MJPG/.avi) unavailable"
    for i in range(n_frames):
        vw.write(np.full((h, w, 3), (i * 5) % 250, np.uint8))
    vw.release()
    return path


def frame_level(frame: np.ndarray) -> int:
    """Recover the synthetic frame index (gray level / 5); robust to +-2 codec error."""
    return int(round(float(frame.mean()) / 5.0))


@pytest.fixture
def synth_video(tmp_path) -> Path:
    return write_video(tmp_path / "v.avi", 40, 20.0)


H2S_HEADER_O = "VIDEO_ID\tVIDEO_NAME\tSENTENCE_ID\tSENTENCE_NAME\tSTART\tEND\tSENTENCE\n"
H2S_HEADER_R = "VIDEO_ID\tVIDEO_NAME\tSENTENCE_ID\tSENTENCE_NAME\tSTART_REALIGNED\tEND_REALIGNED\tSENTENCE\n"


@pytest.fixture
def synth_how2sign(tmp_path) -> Path:
    """Mimics the official How2Sign layout. Sentence texts are synthetic. sentence
    vidA_2 has no clip (like the 2 val rows absent from the official archive)."""
    root = tmp_path / "How2Sign"
    txt = root / "sentence_level/val/text/en/raw_text"
    (txt / "re_aligned").mkdir(parents=True)
    rows = [("vidA", "vidA-1-rgb_front", "vidA_0", "vidA_0-1-rgb_front", 0.0, 1.0, 0.1, 0.9, 'He said "ok" here.'),
            ("vidA", "vidA-1-rgb_front", "vidA_1", "vidA_1-1-rgb_front", 1.0, 2.0, 1.2, 1.9, '"An ""i"" here."'),
            ("vidA", "vidA-1-rgb_front", "vidA_2", "vidA_2-1-rgb_front", 2.0, 3.5, 2.1, 3.4, "No clip, it's missing.")]
    with open(txt / "how2sign_val.csv", "w", encoding="utf-8", newline="") as f:
        f.write(H2S_HEADER_O)
        for r in rows:
            f.write("\t".join(map(str, r[:4] + (r[4], r[5], r[8]))) + "\n")
    with open(txt / "re_aligned/how2sign_realigned_val.csv", "w", encoding="utf-8", newline="") as f:
        f.write(H2S_HEADER_R)
        for r in rows:
            f.write("\t".join(map(str, r[:4] + (r[6], r[7], r[8]))) + "\n")
    clips = root / "sentence_level/val/rgb_front/raw_videos"
    for name, n in (("vidA_0-1-rgb_front", 20), ("vidA_1-1-rgb_front", 20)):
        write_video(clips / f"{name}.avi", n, 20.0).rename(clips / f"{name}.mp4")
    full = root / "video_level/val/rgb_front/raw_videos"
    write_video(full / "vidA-1-rgb_front.avi", 80, 20.0).rename(full / "vidA-1-rgb_front.mp4")
    return root


@pytest.fixture
def synth_vsl_letters(tmp_path) -> Path:
    root = tmp_path / "vsl_data"
    for d in ("A", "DD", "MU"):
        (root / d).mkdir(parents=True)
        for i in range(6):
            cv2.imwrite(str(root / d / f"frame_{i + 1:04d}.jpg"), np.full((24, 32, 3), i * 10, np.uint8))
    return root


@pytest.fixture(scope="session")
def h2s_root() -> Path:
    clips = H2S_ROOT / "sentence_level/val/rgb_front/raw_videos"
    if not clips.exists() or not any(clips.glob("*.mp4")):
        pytest.skip(f"How2Sign subset not found at {H2S_ROOT}")
    return H2S_ROOT
