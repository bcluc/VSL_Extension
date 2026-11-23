"""Dataset-independent interface. Adapters only implement ``_build_records``."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..video.io import FrameSource, ImageFolderReader, VideoReader, VideoUnavailableError, probe_video
from .schema import AnnotationStatus, ManifestRecord, read_manifest, write_manifest


class SampleNotFoundError(KeyError):
    pass


class AnnotationMissingError(LookupError):
    """A requested annotation type (e.g. gloss) does not exist for this sample/dataset."""


@dataclass(frozen=True)
class Annotation:
    sample_id: str
    status: AnnotationStatus
    gloss_sequence: tuple[str, ...] | None
    source_text: str | None
    target_text: str | None
    start_time: float | None
    end_time: float | None

    def require_gloss(self) -> tuple[str, ...]:
        if not self.gloss_sequence:
            raise AnnotationMissingError(f"{self.sample_id}: no gloss annotation (status={self.status.value})")
        return self.gloss_sequence

    def require_translation(self) -> str:
        if not self.target_text:
            raise AnnotationMissingError(f"{self.sample_id}: no translation (status={self.status.value})")
        return self.target_text


@dataclass
class Sample:
    record: ManifestRecord
    annotation: Annotation
    dataset: "SignDataset"

    @property
    def sample_id(self) -> str:
        return self.record.sample_id

    def open_video(self) -> FrameSource:
        return self.dataset.load_video(self.sample_id)


class SignDataset(ABC):
    """Common contract for every sign-language dataset.

    load_manifest()            -> list[ManifestRecord] (normalized, cached)
    get_sample(sample_id)      -> Sample (record + annotation + lazy video handle)
    load_video(sample_id)      -> FrameSource (streaming; never decodes everything up front)
    load_annotations(sample_id)-> Annotation
    get_metadata(sample_id)    -> dict (record, provenance, probed video info)
    """

    dataset_name: str = "abstract"
    language: str = ""
    license: str | None = None

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        self._records: list[ManifestRecord] | None = None
        self._index: dict[str, ManifestRecord] = {}
        self._probe_cache: dict[str, Any] = {}
        self.build_warnings: list[dict[str, Any]] = []

    # --------------------------------------------------------------- to implement
    @abstractmethod
    def _build_records(self) -> list[ManifestRecord]:
        """Parse the dataset's native annotation files into ManifestRecords."""

    def _open_source(self, record: ManifestRecord) -> FrameSource:
        path = Path(record.video_path)
        if not path.exists():
            raise VideoUnavailableError(f"{record.sample_id}: video not available locally ({path})")
        if path.is_dir():
            return ImageFolderReader(path, fps=record.fps)
        return VideoReader(path, record.start_time, record.end_time)

    # --------------------------------------------------------------- public API
    def load_manifest(self, refresh: bool = False) -> list[ManifestRecord]:
        if self._records is None or refresh:
            self.build_warnings = []
            records = self._build_records()
            index: dict[str, ManifestRecord] = {}
            for r in records:
                if r.sample_id in index:
                    raise ValueError(f"duplicate sample_id {r.sample_id}")
                index[r.sample_id] = r
            self._records, self._index = records, index
        return self._records

    def sample_ids(self, available_only: bool = False) -> list[str]:
        return [r.sample_id for r in self.load_manifest() if not available_only or r.video_available]

    def _record(self, sample_id: str) -> ManifestRecord:
        self.load_manifest()
        try:
            return self._index[sample_id]
        except KeyError:
            raise SampleNotFoundError(sample_id) from None

    def load_annotations(self, sample_id: str) -> Annotation:
        r = self._record(sample_id)
        return Annotation(r.sample_id, AnnotationStatus(r.annotation_status),
                          tuple(r.gloss_sequence) if r.gloss_sequence else None,
                          r.source_text, r.target_text, r.start_time, r.end_time)

    def load_video(self, sample_id: str) -> FrameSource:
        return self._open_source(self._record(sample_id))

    def get_sample(self, sample_id: str) -> Sample:
        return Sample(self._record(sample_id), self.load_annotations(sample_id), self)

    def get_metadata(self, sample_id: str) -> dict[str, Any]:
        r = self._record(sample_id)
        meta: dict[str, Any] = {"record": r.to_dict(), "license": self.license,
                                "dataset_name": self.dataset_name, "video": None}
        p = Path(r.video_path)
        if p.is_file():
            if r.video_path not in self._probe_cache:
                self._probe_cache[r.video_path] = probe_video(p).to_dict()
            meta["video"] = self._probe_cache[r.video_path]
        elif p.is_dir():
            meta["video"] = ImageFolderReader(p, fps=r.fps).info.to_dict()
        return meta

    def save_manifest(self, path: str | Path) -> Path:
        return write_manifest(self.load_manifest(), path)

    def __len__(self) -> int:
        return len(self.load_manifest())

    def __iter__(self) -> Iterator[Sample]:
        for r in self.load_manifest():
            yield self.get_sample(r.sample_id)

    # --------------------------------------------------------------- helpers
    def _warn(self, kind: str, **info: Any) -> None:
        self.build_warnings.append({"kind": kind, **info})

    def _probe_fps(self, path: Path) -> float | None:
        """Measured FPS of an existing file, else None (never guessed)."""
        key = str(path)
        if key not in self._probe_cache:
            try:
                self._probe_cache[key] = probe_video(path).to_dict()
            except VideoUnavailableError:
                self._probe_cache[key] = None
        info = self._probe_cache[key]
        return None if info is None else info["fps"]


class ManifestDataset(SignDataset):
    """Generic dataset backed by an existing common-schema JSONL manifest."""

    def __init__(self, manifest_path: str | Path, split: str | None = None):
        self.manifest_path = Path(manifest_path)
        super().__init__(self.manifest_path.parent, split or "all")

    def _build_records(self) -> list[ManifestRecord]:
        recs = read_manifest(self.manifest_path)
        return [r for r in recs if self.split == "all" or r.split == self.split]
