"""Per-frame body + hand landmarks for isolated sign recognition, and the feature vector.

The same functions run offline (dataset clips) and live (webcam), so training and the demo
see identical features.

Raw landmarks per frame (MediaPipe Tasks, IMAGE mode):
    pose   (33, 4)  x, y, z, visibility    PoseLandmarker (lite)
    hands  (2, 21, 3) x, y, z              HandLandmarker, slot 0 = the person's left hand,
                                           slot 1 = right hand (matched to the pose wrists)
    hand_present (2,) bool

``sequence_features`` turns a raw sequence into a fixed (T, D) array:
* crop to the active span (frames with a visible hand), resample to ``T`` frames;
* normalise coordinates to the signer's body: origin = shoulder midpoint,
  unit = shoulder width (medians over the sequence);
* hand shape relative to the wrist, scaled by hand size;
* first differences of the body-relative coordinates (motion).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

POSE_IDX = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # nose, eyes, ears, mouth, shoulders, elbows, wrists
# left/right pairs inside POSE_IDX order (for mirroring): eyes, ears, mouth, shoulders, elbows, wrists
_POSE_SWAP = {2: 5, 5: 2, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
T_DEFAULT = 32


@dataclass
class RawSequence:
    pose: np.ndarray          # (N, 33, 4)
    hands: np.ndarray         # (N, 2, 21, 3)
    hand_present: np.ndarray  # (N, 2) bool
    pose_present: np.ndarray  # (N,) bool

    def __len__(self) -> int:
        return len(self.pose)

    def save(self, path: Path, **meta) -> None:
        np.savez_compressed(path, pose=self.pose, hands=self.hands, hand_present=self.hand_present,
                            pose_present=self.pose_present, **meta)

    @classmethod
    def load(cls, path: Path) -> "RawSequence":
        z = np.load(path)
        return cls(z["pose"], z["hands"], z["hand_present"], z["pose_present"])

    @classmethod
    def stack(cls, frames: list[tuple]) -> "RawSequence":
        if not frames:
            return cls(np.zeros((0, 33, 4), np.float32), np.zeros((0, 2, 21, 3), np.float32),
                       np.zeros((0, 2), bool), np.zeros((0,), bool))
        p, h, hp, pp = zip(*frames)
        return cls(np.stack(p), np.stack(h), np.stack(hp), np.array(pp))


class LandmarkExtractor:
    """Pose + two hands per BGR frame."""

    def __init__(self, pose_model: str | Path, hand_model: str | Path, min_conf: float = 0.5,
                 mode: str = "image", parallel: bool = False):
        """``mode="image"`` detects from scratch on every frame (how the training features were
        made). ``mode="video"`` tracks landmarks across frames (faster live; needs increasing
        timestamps). ``parallel=True`` runs pose and hands in two threads (same output, lower
        latency; leave it off when many extraction processes already share the CPU)."""
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision as mpv
        if mode not in ("image", "video"):
            raise ValueError(f"mode must be 'image' or 'video', not {mode!r}")
        self._mp = mp
        self.mode = mode
        self._last_ms = -1
        for p in (pose_model, hand_model):
            if not Path(p).exists():
                raise FileNotFoundError(f"MediaPipe model not found: {p}")
        rm = mpv.RunningMode.VIDEO if mode == "video" else mpv.RunningMode.IMAGE
        self._pose = mpv.PoseLandmarker.create_from_options(mpv.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(pose_model)),
            running_mode=rm, num_poses=1,
            min_pose_detection_confidence=min_conf, min_pose_presence_confidence=min_conf))
        self._hands = mpv.HandLandmarker.create_from_options(mpv.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=rm, num_hands=2,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf))
        self._pool = None
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mp-hands")

    def frame(self, bgr: np.ndarray, timestamp_ms: int | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            ms = self._last_ms + 33 if timestamp_ms is None else int(timestamp_ms)
            ms = max(ms, self._last_ms + 1)  # MediaPipe needs strictly increasing timestamps
            self._last_ms = ms
            detect_pose = lambda: self._pose.detect_for_video(img, ms)  # noqa: E731
            detect_hands = lambda: self._hands.detect_for_video(img, ms)  # noqa: E731
        else:
            detect_pose = lambda: self._pose.detect(img)  # noqa: E731
            detect_hands = lambda: self._hands.detect(img)  # noqa: E731
        fut = self._pool.submit(detect_hands) if self._pool else None
        pose = np.zeros((33, 4), np.float32)
        pres = detect_pose()
        pose_ok = bool(pres.pose_landmarks)
        if pose_ok:
            pose[:] = [(l.x, l.y, l.z, l.visibility) for l in pres.pose_landmarks[0]]
        hands = np.zeros((2, 21, 3), np.float32)
        present = np.zeros(2, bool)
        hres = fut.result() if fut else detect_hands()
        cands = [np.array([(l.x, l.y, l.z) for l in lms], np.float32) for lms in hres.hand_landmarks]
        labels = [h[0].category_name for h in hres.handedness]
        for slot, kp in _assign_slots(cands, labels, pose if pose_ok else None):
            hands[slot] = kp
            present[slot] = True
        return pose, hands, present, pose_ok

    def video(self, path: str | Path, max_frames: int = 400) -> tuple[RawSequence, float]:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        step_ms = 1000.0 / fps if fps > 0 else 33.0
        base = self._last_ms + 1000  # video mode: keep timestamps increasing across clips
        frames = []
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(self.frame(bgr, int(base + len(frames) * step_ms)))
        cap.release()
        return RawSequence.stack(frames), fps

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown()
        self._pose.close()
        self._hands.close()


def _assign_slots(cands: list[np.ndarray], labels: list[str], pose: np.ndarray | None):
    """Match detected hands to the person's left/right using the pose wrists when available."""
    if not cands:
        return []
    if pose is not None:
        wrists = pose[[POSE_LEFT_WRIST, POSE_RIGHT_WRIST], :2]
        if len(cands) == 1:
            d = np.linalg.norm(wrists - cands[0][0, :2], axis=1)
            return [(int(np.argmin(d)), cands[0])]
        a, b = cands[0][0, :2], cands[1][0, :2]
        keep = (np.linalg.norm(wrists[0] - a) + np.linalg.norm(wrists[1] - b)
                <= np.linalg.norm(wrists[0] - b) + np.linalg.norm(wrists[1] - a))
        return [(0, cands[0]), (1, cands[1])] if keep else [(0, cands[1]), (1, cands[0])]
    # No pose: MediaPipe assumes a mirrored (selfie) image; on an unmirrored frame its
    # "Left" label is the person's right hand.
    out, used = [], set()
    for kp, lab in zip(cands, labels):
        slot = 1 if lab == "Left" else 0
        if slot not in used:
            out.append((slot, kp))
            used.add(slot)
    return out


# ---------------------------------------------------------------------- features
def active_span(hand_present: np.ndarray, margin: int = 2) -> tuple[int, int]:
    """[start, end) of frames where at least one hand is visible; whole clip if none."""
    idx = np.flatnonzero(hand_present.any(axis=1))
    if len(idx) == 0:
        return 0, len(hand_present)
    return max(0, idx[0] - margin), min(len(hand_present), idx[-1] + 1 + margin)


def mirror(seq: RawSequence) -> RawSequence:
    """Horizontal flip: x -> 1 - x, swap left/right body landmarks and hand slots."""
    pose = seq.pose.copy()
    pose[..., 0] = np.where(seq.pose_present[:, None], 1.0 - pose[..., 0], 0.0)
    perm = list(range(33))
    for a, b in _POSE_SWAP.items():
        perm[a] = b
    pose = pose[:, perm]
    hands = seq.hands[:, ::-1].copy()
    hp = seq.hand_present[:, ::-1].copy()
    hands[..., 0] = np.where(hp[..., None], 1.0 - hands[..., 0], 0.0)
    return RawSequence(pose, hands, hp, seq.pose_present.copy())


def _resample_idx(n: int, t: int) -> np.ndarray:
    return np.clip(np.round(np.linspace(0, n - 1, t)).astype(int), 0, max(n - 1, 0))


def sequence_features(seq: RawSequence, t: int = T_DEFAULT, span: tuple[int, int] | None = None) -> np.ndarray:
    """Raw landmark sequence -> (t, D) float32 feature array."""
    if len(seq) == 0:
        raise ValueError("empty sequence")
    s, e = span if span is not None else active_span(seq.hand_present)
    idx = np.arange(s, e)[_resample_idx(e - s, t)]
    pose, hands, hp, pp = seq.pose[idx], seq.hands[idx], seq.hand_present[idx], seq.pose_present[idx]

    # body frame from the whole clip (robust to a few missing poses)
    all_pp = seq.pose_present
    if all_pp.any():
        sh = seq.pose[all_pp][:, [11, 12], :2]
        center = np.median(sh.mean(axis=1), axis=0)
        scale = float(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1)))
    else:  # fall back to the image frame
        center, scale = np.array([0.5, 0.5], np.float32), 0.3
    scale = max(scale, 1e-3)

    body = (pose[:, POSE_IDX, :2] - center) / scale * pp[:, None, None]       # (t, 13, 2)
    hglob = (hands[..., :2] - center) / scale * hp[..., None, None]            # (t, 2, 21, 2)
    rel = hands - hands[:, :, :1]                                              # wrist-relative
    size = np.linalg.norm(rel[..., :2], axis=-1).max(axis=-1, keepdims=True)  # (t, 2, 1)
    shape = rel / np.maximum(size, 1e-3)[..., None] * hp[..., None, None]      # (t, 2, 21, 3)

    coords = np.concatenate([body.reshape(t, -1), hglob.reshape(t, -1)], axis=1)
    motion = np.diff(coords, axis=0, prepend=coords[:1])
    feats = np.concatenate([coords, shape.reshape(t, -1), hp.astype(np.float32), motion], axis=1)
    return feats.astype(np.float32)


def feature_dim(t: int = T_DEFAULT) -> int:
    dummy = RawSequence(np.zeros((2, 33, 4), np.float32), np.zeros((2, 2, 21, 3), np.float32),
                        np.ones((2, 2), bool), np.ones(2, bool))
    return sequence_features(dummy, t).shape[1]
