"""Common manifest schema shared by every dataset adapter.

Rules
-----
* Missing information is ``None`` (JSON ``null``). Adapters must never invent values.
* ``video_path`` is always the path the sample's video is *expected* at. Whether it
  exists locally is recorded in ``meta["video_available"]``.
* ``start_time`` / ``end_time`` are seconds **relative to ``video_path``**. ``None``
  means "the whole file".
* ``gloss_sequence`` is only filled from real gloss annotations of the same sign
  language. Translations are never converted into glosses.
* ``meta`` holds dataset-specific provenance (ids, original timestamps, notes).
  Model code must not depend on it.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class AnnotationStatus(str, Enum):
    """What kind of supervision a sample actually carries."""

    GLOSS_AND_TRANSLATION = "gloss_and_translation"  # CSLR + SLT supervision
    GLOSS_ONLY = "gloss_only"                        # CSLR supervision only
    TRANSLATION_ONLY = "translation_only"            # sentence-level SLT only (How2Sign today)
    ISOLATED_LABEL = "isolated_label"                # one label per clip (e.g. fingerspelling)
    UNANNOTATED = "unannotated"

    @classmethod
    def infer(cls, gloss: list[str] | None, target_text: str | None,
              isolated: bool = False) -> "AnnotationStatus":
        if isolated and gloss:
            return cls.ISOLATED_LABEL
        if gloss and target_text:
            return cls.GLOSS_AND_TRANSLATION
        if gloss:
            return cls.GLOSS_ONLY
        if target_text:
            return cls.TRANSLATION_ONLY
        return cls.UNANNOTATED


# Project-level sign-language ids. ISO 639-3: ASL = "ase". Vietnamese sign languages
# have regional ISO codes (hab = Hanoi, haf = Haiphong, hos = Ho Chi Minh City);
# NOTE: ISO "vsl" is *Venezuelan* Sign Language, so it is never used as a code here.
SIGN_LANGUAGES = {
    "asl": {"name": "American Sign Language", "iso639_3": "ase"},
    "dgs": {"name": "German Sign Language (Deutsche Gebärdensprache)", "iso639_3": "gsg"},
    "vsl": {"name": "Vietnamese Sign Language", "iso639_3": None,
            "note": "regional ISO codes: hab, haf, hos"},
}


@dataclass
class ManifestRecord:
    sample_id: str
    dataset_name: str
    language: str
    video_path: str
    split: str
    annotation_status: str
    signer_id: str | None = None
    gloss_sequence: list[str] | None = None
    source_text: str | None = None
    target_text: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    fps: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.annotation_status, AnnotationStatus):
            self.annotation_status = self.annotation_status.value
        AnnotationStatus(self.annotation_status)  # validate
        if self.language not in SIGN_LANGUAGES:
            raise ValueError(f"unknown sign language id {self.language!r}")
        if self.start_time is not None and self.end_time is not None \
                and self.end_time <= self.start_time:
            raise ValueError(f"{self.sample_id}: end_time {self.end_time} <= start_time {self.start_time}")
        if self.gloss_sequence is not None and not all(isinstance(g, str) and g for g in self.gloss_sequence):
            raise ValueError(f"{self.sample_id}: gloss_sequence must be non-empty strings")

    @property
    def has_gloss(self) -> bool:
        return bool(self.gloss_sequence)

    @property
    def has_translation(self) -> bool:
        return bool(self.target_text)

    @property
    def video_available(self) -> bool:
        return bool(self.meta.get("video_available", Path(self.video_path).exists()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ManifestRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


MANIFEST_FIELDS = [f.name for f in fields(ManifestRecord)]


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> Path:
    """Write JSON Lines (one record per line, ``null`` for missing values)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    with open(path, encoding="utf-8") as f:
        return [ManifestRecord.from_dict(json.loads(line)) for line in f if line.strip()]


def write_manifest_csv(records: Iterable[ManifestRecord], path: str | Path) -> Path:
    """Flat CSV view for inspection (gloss joined by spaces, meta as JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in records:
            d = r.to_dict()
            d["gloss_sequence"] = None if r.gloss_sequence is None else " ".join(r.gloss_sequence)
            d["meta"] = json.dumps(r.meta, ensure_ascii=False)
            w.writerow({k: ("" if v is None else v) for k, v in d.items()})
    return path
