"""Variable-length batching. Produces numpy arrays; ``Batch.to_torch()`` converts.

Missing supervision is explicit: per-sample ``has_gloss`` / ``has_text`` flags and
zero lengths, never placeholder labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..text.vocab import GlossVocabulary, TextVocabulary
from .base import AnnotationMissingError
from .schema import ManifestRecord


def pad_sequences(seqs: Sequence[np.ndarray], pad_value: float = 0.0,
                  max_len: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad arrays of shape (T_i, ...) -> (B, T_max, ...), lengths (B,), mask (B, T_max)."""
    if not seqs:
        raise ValueError("empty batch")
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    t_max = int(max_len or lengths.max())
    tail = seqs[0].shape[1:]
    for s in seqs:
        if s.shape[1:] != tail:
            raise ValueError(f"inconsistent feature shapes {s.shape[1:]} vs {tail}")
    out = np.full((len(seqs), t_max) + tail, pad_value, dtype=seqs[0].dtype)
    mask = np.zeros((len(seqs), t_max), dtype=bool)
    for i, s in enumerate(seqs):
        n = min(len(s), t_max)
        out[i, :n] = s[:n]
        mask[i, :n] = True
    return out, np.minimum(lengths, t_max), mask


@dataclass
class Batch:
    sample_ids: list[str]
    inputs: np.ndarray                 # (B, T, ...) frames or keypoints
    input_mask: np.ndarray             # (B, T) True = real timestep
    input_lengths: np.ndarray          # (B,)
    frame_valid: np.ndarray            # (B, T) True = decoded frame (False = missing/padding)
    gloss_ids: np.ndarray | None = None      # (B, Lg) padded with gloss pad id
    gloss_lengths: np.ndarray | None = None  # (B,) 0 where no gloss
    has_gloss: np.ndarray | None = None      # (B,) bool
    text_ids: np.ndarray | None = None       # (B, Lt) with <bos>/<eos>, padded
    text_lengths: np.ndarray | None = None
    has_text: np.ndarray | None = None
    annotation_status: list[str] = field(default_factory=list)

    def to_torch(self) -> dict[str, Any]:
        import torch
        out: dict[str, Any] = {"sample_ids": self.sample_ids, "annotation_status": self.annotation_status}
        for k in ("inputs", "input_mask", "input_lengths", "frame_valid", "gloss_ids", "gloss_lengths",
                  "has_gloss", "text_ids", "text_lengths", "has_text"):
            v = getattr(self, k)
            out[k] = None if v is None else torch.from_numpy(np.ascontiguousarray(v))
        return out


def collate(items: Sequence[dict[str, Any]], gloss_vocab: GlossVocabulary | None = None,
            text_vocab: TextVocabulary | None = None, require_gloss: bool = False,
            require_text: bool = False) -> Batch:
    """items: dicts with ``record`` (ManifestRecord), ``inputs`` (T, ...) array and
    optional ``valid_mask`` (T,)."""
    recs: list[ManifestRecord] = [it["record"] for it in items]
    inputs, lengths, mask = pad_sequences([it["inputs"] for it in items])
    valid = np.zeros_like(mask)
    for i, it in enumerate(items):
        vm = it.get("valid_mask")
        n = lengths[i]
        valid[i, :n] = True if vm is None else vm[:n]
    b = Batch([r.sample_id for r in recs], inputs, mask, lengths, valid,
              annotation_status=[r.annotation_status for r in recs])

    if gloss_vocab is not None or require_gloss:
        missing = [r.sample_id for r in recs if not r.has_gloss]
        if require_gloss and missing:
            raise AnnotationMissingError(f"gloss required but missing for {missing}")
        if gloss_vocab is not None:
            seqs = [np.array(gloss_vocab.encode(r.gloss_sequence or []), dtype=np.int64) for r in recs]
            b.gloss_ids, b.gloss_lengths, _ = _pad_ids(seqs, gloss_vocab.pad_id)
            b.has_gloss = np.array([r.has_gloss for r in recs])

    if text_vocab is not None or require_text:
        missing = [r.sample_id for r in recs if not r.has_translation]
        if require_text and missing:
            raise AnnotationMissingError(f"translation required but missing for {missing}")
        if text_vocab is not None:
            seqs = [np.array(text_vocab.encode_text(r.target_text) if r.target_text else [], dtype=np.int64)
                    for r in recs]
            b.text_ids, b.text_lengths, _ = _pad_ids(seqs, text_vocab.pad_id)
            b.has_text = np.array([r.has_translation for r in recs])
    return b


def _pad_ids(seqs: list[np.ndarray], pad_id: int):
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    out = np.full((len(seqs), max(int(lengths.max()), 1)), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out, lengths, out != pad_id
