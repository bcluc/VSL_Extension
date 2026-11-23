"""Vietnamese Sign Language (VSL) adapter.

Two layouts:

``letter_frames`` (IMPLEMENTED, tested on synthetic fixtures)
    ``<root>/<LABEL_ID>/frame_XXXX.jpg`` (or ``.png``): one folder per fingerspelling letter,
    named by the ASCII label id of ``signbridge.letters.labels`` (``DD`` = Đ, ``MU`` = mũ,
    ``RAU`` = râu; other names are used as they are). No FPS, signer or split information is
    stored, so those fields are None / "unsplit".
    This is ISOLATED recognition data: one label per sequence -> status ``isolated_label``.
    The letter demo itself records landmarks, not images (``signbridge.letters.recordings``).

``continuous_csv`` (PROPOSED schema, no real continuous VSL data exists yet)
    ``<root>/manifest.csv`` with columns::

        sample_id, video_path, split, signer_id, gloss, vietnamese_text, start_time, end_time

    ``gloss`` = space-separated VSL glosses (may be empty), ``vietnamese_text`` = target
    sentence (may be empty), times in seconds within ``video_path`` (may be empty =
    whole file). Empty cells become None. This reader is tested only on synthetic
    fixtures; adjust it once the real VSL annotation format has been inspected.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..letters.labels import label_from_dirname
from .base import SignDataset
from .schema import AnnotationStatus, ManifestRecord

CONTINUOUS_COLUMNS = ["sample_id", "video_path", "split", "signer_id", "gloss",
                      "vietnamese_text", "start_time", "end_time"]


class VSLDataset(SignDataset):
    dataset_name = "vsl"
    language = "vsl"
    license = "project-internal (SignBridge); see repository owner"

    def __init__(self, root: str | Path, split: str = "unsplit", layout: str = "letter_frames",
                 known_labels: set[str] | None = None):
        if layout not in ("letter_frames", "continuous_csv"):
            raise ValueError("layout must be 'letter_frames' or 'continuous_csv'")
        super().__init__(root, split)
        self.layout = layout
        self.known_labels = known_labels

    def _build_records(self) -> list[ManifestRecord]:
        return self._letter_frames() if self.layout == "letter_frames" else self._continuous()

    # ------------------------------------------------------------ letter_frames
    def _letter_frames(self) -> list[ManifestRecord]:
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        records = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            label = label_from_dirname(d.name)
            if self.known_labels is not None and d.name not in self.known_labels:
                self._warn("unknown_label", folder=d.name, label=label)
            frames = sorted([*d.glob("frame_*.jpg"), *d.glob("frame_*.png")])
            if not frames:
                self._warn("empty_folder", folder=d.name)
                continue
            records.append(ManifestRecord(
                sample_id=f"letters/{d.name}", dataset_name=self.dataset_name, language=self.language,
                video_path=str(d), split=self.split,
                annotation_status=AnnotationStatus.ISOLATED_LABEL,
                signer_id=None, gloss_sequence=[label], source_text=None, target_text=None,
                start_time=None, end_time=None, fps=None,
                meta={"layout": "letter_frames", "label": label, "n_frames": len(frames),
                      "video_available": True,
                      "note": "isolated fingerspelling; gloss_sequence holds the single letter label"}))
        return records

    # ------------------------------------------------------------ continuous_csv
    def _continuous(self) -> list[ManifestRecord]:
        path = self.root / "manifest.csv"
        if not path.exists():
            raise FileNotFoundError(f"continuous VSL manifest not found: {path}")
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing = set(CONTINUOUS_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            rows = list(reader)
        records = []
        for r in rows:
            split = (r["split"] or "").strip() or "unsplit"
            if self.split not in ("all", split):
                continue
            gloss = (r["gloss"] or "").split() or None
            text = (r["vietnamese_text"] or "").strip() or None
            vp = Path(r["video_path"])
            vp = vp if vp.is_absolute() else self.root / vp
            records.append(ManifestRecord(
                sample_id=r["sample_id"], dataset_name=self.dataset_name, language=self.language,
                video_path=str(vp), split=split,
                annotation_status=AnnotationStatus.infer(gloss, text),
                signer_id=(r["signer_id"] or "").strip() or None, gloss_sequence=gloss,
                source_text=None, target_text=text,
                start_time=_f(r["start_time"]), end_time=_f(r["end_time"]),
                fps=self._probe_fps(vp) if vp.is_file() else None,
                meta={"layout": "continuous_csv", "target_language": "vi",
                      "video_available": vp.exists()}))
        return records


def _f(v: str | None) -> float | None:
    v = (v or "").strip()
    return float(v) if v else None
