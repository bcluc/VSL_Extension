"""Model interfaces for Continuous Sign Language Translation.

    frames (B,T,C,H,W) | keypoints (B,T,K,C)
        -> VisualEncoder      -> SequenceFeatures (B,T,D)        per-frame features
        -> TemporalEncoder    -> SequenceFeatures (B,T',D')      context, optional downsampling
        -> GlossDecoder (CTC) -> GlossOutput (T',B,V) log-probs  [needs gloss labels to train]
        -> TranslationModel   -> logits (B,L,V) / token ids       [needs target text to train]

Every stage passes explicit ``lengths`` + ``mask`` so variable-length sequences are
never silently mixed with padding. Any implementation honouring these contracts can
be swapped in (RGB CNN, pose MLP/GCN, video transformer, pretrained LLM decoder ...).

Translation can consume either the temporal features directly (gloss-free
Sign2Text - the only trainable option on How2Sign, which has no glosses) or the
gloss decoder's representation (Sign2Gloss2Text - needs gloss labels, e.g. future VSL).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SequenceFeatures:
    features: torch.Tensor  # (B, T, D)
    lengths: torch.Tensor   # (B,) int64, number of valid steps
    mask: torch.Tensor      # (B, T) bool, True = valid

    def __post_init__(self) -> None:
        B, T = self.features.shape[:2]
        if self.lengths.shape != (B,) or self.mask.shape != (B, T):
            raise ValueError(f"inconsistent shapes features={tuple(self.features.shape)} "
                             f"lengths={tuple(self.lengths.shape)} mask={tuple(self.mask.shape)}")
        if int(self.lengths.max()) > T:
            raise ValueError("lengths exceed time dimension")

    @staticmethod
    def mask_from_lengths(lengths: torch.Tensor, T: int) -> torch.Tensor:
        return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


@dataclass
class GlossOutput:
    log_probs: torch.Tensor  # (T, B, V) log-softmax, CTC layout
    lengths: torch.Tensor    # (B,)
    blank_id: int


class VisualEncoder(nn.Module, ABC):
    """Per-timestep spatial encoder. ``modality`` is 'rgb' or 'pose'."""

    modality: str = "rgb"
    output_dim: int

    @abstractmethod
    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> SequenceFeatures:
        """rgb: (B,T,C,H,W) float; pose: (B,T,K,C) float -> (B,T,output_dim)."""


class TemporalEncoder(nn.Module, ABC):
    output_dim: int
    downsample: int = 1  # T' = ceil(T / downsample)

    @abstractmethod
    def forward(self, x: SequenceFeatures) -> SequenceFeatures: ...


class GlossDecoder(nn.Module, ABC):
    """CTC-style gloss recognizer head (CSLR)."""

    vocab_size: int
    blank_id: int = 0

    @abstractmethod
    def forward(self, x: SequenceFeatures) -> GlossOutput: ...

    @abstractmethod
    def loss(self, out: GlossOutput, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, out: GlossOutput) -> list[list[int]]: ...


class TranslationModel(nn.Module, ABC):
    """Sequence-to-sequence text generator conditioned on sign features."""

    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int

    @abstractmethod
    def forward(self, memory: SequenceFeatures, target_in: torch.Tensor) -> torch.Tensor:
        """Teacher forcing. target_in (B,L) starts with <bos>; returns logits (B,L,V)."""

    @abstractmethod
    def generate(self, memory: SequenceFeatures, max_len: int = 50) -> list[list[int]]: ...


class SignTranslationPipeline(nn.Module):
    """Composable CSLT pipeline. ``gloss_decoder`` and ``translator`` are optional so the
    same graph serves CSLR-only, gloss-free SLT, or Sign2Gloss2Text."""

    def __init__(self, visual: VisualEncoder, temporal: TemporalEncoder,
                 gloss_decoder: GlossDecoder | None = None, translator: TranslationModel | None = None):
        super().__init__()
        if temporal_in := getattr(temporal, "input_dim", None):
            if temporal_in != visual.output_dim:
                raise ValueError(f"temporal.input_dim {temporal_in} != visual.output_dim {visual.output_dim}")
        self.visual, self.temporal = visual, temporal
        self.gloss_decoder, self.translator = gloss_decoder, translator

    def encode(self, inputs: torch.Tensor, lengths: torch.Tensor) -> SequenceFeatures:
        return self.temporal(self.visual(inputs, lengths))

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor,
                target_in: torch.Tensor | None = None) -> dict:
        enc = self.encode(inputs, lengths)
        out: dict = {"encoder": enc}
        if self.gloss_decoder is not None:
            out["gloss"] = self.gloss_decoder(enc)
        if self.translator is not None and target_in is not None:
            out["translation_logits"] = self.translator(enc, target_in)
        return out
