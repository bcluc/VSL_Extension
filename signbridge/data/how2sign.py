"""How2Sign (ASL <-> English) adapter. Used ONLY to validate the pipeline.

Native layout created by the official download script (and by
``scripts/fetch_how2sign_subset.py``)::

    <root>/sentence_level/<split>/text/en/raw_text/how2sign_<split>.csv                    original timestamps
    <root>/sentence_level/<split>/text/en/raw_text/re_aligned/how2sign_realigned_<split>.csv manual re-alignment
    <root>/sentence_level/<split>/rgb_front/raw_videos/<SENTENCE_NAME>.mp4                 sentence clips
    <root>/video_level/<split>/rgb_front/raw_videos/<VIDEO_NAME>.mp4                       full videos

CSV files are TAB-separated (standard CSV quoting) with columns
``VIDEO_ID VIDEO_NAME SENTENCE_ID SENTENCE_NAME START END SENTENCE`` (original) or
``... START_REALIGNED END_REALIGNED SENTENCE`` (re-aligned). Times are seconds on the
*full video* timeline.

Facts this adapter encodes (verified in docs/dataset_audit.md):
* There are **no gloss annotations** in the public release -> ``gloss_sequence`` is None.
* ``SENTENCE`` is an English translation, stored as ``target_text`` (never as gloss).
* Sentence clips were cut with the ORIGINAL timestamps (official website), so the
  ``clips`` view pairs each clip with the original row; re-aligned boundaries are only
  usable on full videos (``realigned_full`` view).
* The meaning of the numeric suffix in ``VIDEO_NAME`` (e.g. ``-3-rgb_front``) is not
  documented in the downloaded files, so ``signer_id`` stays None.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .base import SignDataset
from .schema import AnnotationStatus, ManifestRecord

VIEWS = ("clips", "original_full", "realigned_full")
ORIGINAL_COLUMNS = ["VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME", "START", "END", "SENTENCE"]
REALIGNED_COLUMNS = ["VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME",
                     "START_REALIGNED", "END_REALIGNED", "SENTENCE"]


def read_how2sign_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    """Read a How2Sign TSV with standard CSV quoting. The audit found 8 val sentences
    stored as quoted fields with doubled quotes (e.g. ``"... an ""i"" you're ..."``);
    standard quoting decodes them to ``... an "i" you're ...``. Unquoted fields keep
    apostrophes/quotes literally."""
    if not path.exists():
        raise FileNotFoundError(f"How2Sign annotation file missing: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != expected_columns:
            raise ValueError(f"{path.name}: unexpected columns {reader.fieldnames}, expected {expected_columns}")
        rows = list(reader)
    for i, r in enumerate(rows):
        if None in r or any(v is None for v in r.values()):
            raise ValueError(f"{path.name}: malformed row {i + 2} (wrong number of fields)")
    return rows


class How2SignDataset(SignDataset):
    dataset_name = "how2sign"
    language = "asl"
    license = "CC BY-NC 4.0 (non-commercial research only)"

    def __init__(self, root: str | Path, split: str = "val", view: str = "clips",
                 probe_videos: bool = True):
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        super().__init__(root, split)
        self.view = view
        self.probe_videos = probe_videos

    # paths ------------------------------------------------------------------
    @property
    def text_dir(self) -> Path:
        return self.root / "sentence_level" / self.split / "text" / "en" / "raw_text"

    @property
    def original_csv(self) -> Path:
        return self.text_dir / f"how2sign_{self.split}.csv"

    @property
    def realigned_csv(self) -> Path:
        return self.text_dir / "re_aligned" / f"how2sign_realigned_{self.split}.csv"

    def clip_path(self, sentence_name: str) -> Path:
        return self.root / "sentence_level" / self.split / "rgb_front" / "raw_videos" / f"{sentence_name}.mp4"

    def video_path(self, video_name: str) -> Path:
        return self.root / "video_level" / self.split / "rgb_front" / "raw_videos" / f"{video_name}.mp4"

    # parsing ----------------------------------------------------------------
    def _build_records(self) -> list[ManifestRecord]:
        orig = {r["SENTENCE_NAME"]: r for r in read_how2sign_csv(self.original_csv, ORIGINAL_COLUMNS)}
        realigned: dict[str, dict[str, str]] = {}
        if self.realigned_csv.exists():
            realigned = {r["SENTENCE_NAME"]: r for r in read_how2sign_csv(self.realigned_csv, REALIGNED_COLUMNS)}
        elif self.view == "realigned_full":
            raise FileNotFoundError(self.realigned_csv)

        names = list(realigned) if self.view == "realigned_full" else list(orig)
        records: list[ManifestRecord] = []
        for name in names:
            o, ra = orig.get(name), realigned.get(name)
            row = ra if self.view == "realigned_full" else o
            if row is None:
                self._warn("missing_in_original_csv", sample=name)
                continue
            text = (row["SENTENCE"] or "").strip() or None
            if o and ra and o["SENTENCE"] != ra["SENTENCE"]:
                self._warn("sentence_text_differs_between_csvs", sample=name)
            meta = {
                "video_id": row["VIDEO_ID"], "video_name": row["VIDEO_NAME"],
                "sentence_id": row["SENTENCE_ID"], "sentence_name": name,
                "target_language": "en", "view": self.view,
                "original_start": _f(o["START"]) if o else None,
                "original_end": _f(o["END"]) if o else None,
                "realigned_start": _f(ra["START_REALIGNED"]) if ra else None,
                "realigned_end": _f(ra["END_REALIGNED"]) if ra else None,
            }
            if self.view == "clips":
                path, start, end = self.clip_path(name), None, None
                sid = name
            else:
                path = self.video_path(row["VIDEO_NAME"])
                key = "original" if self.view == "original_full" else "realigned"
                start, end = meta[f"{key}_start"], meta[f"{key}_end"]
                sid = f"{name}#{key}"
                if start is None or end is None or end <= start:
                    self._warn("invalid_interval", sample=name, start=start, end=end)
                    continue
            meta["video_available"] = path.exists()
            fps = self._probe_fps(path) if (self.probe_videos and meta["video_available"]) else None
            records.append(ManifestRecord(
                sample_id=sid, dataset_name=self.dataset_name, language=self.language,
                video_path=str(path), split=self.split,
                annotation_status=AnnotationStatus.infer(None, text),
                signer_id=None, gloss_sequence=None, source_text=None, target_text=text,
                start_time=start, end_time=end, fps=fps, meta=meta))
        return records


def _f(v: str | None) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    return float(v)
