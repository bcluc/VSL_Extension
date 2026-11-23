"""Small, randomly initialised reference implementations of the interfaces.

They exist to validate tensor contracts end-to-end (shapes, masks, lengths, CTC loss
computability, autoregressive decoding loop). They are NOT trained and their outputs
carry no linguistic meaning. Production choices (e.g. pretrained video backbones,
pose GCNs, mBART/LLM decoders) plug in behind the same interfaces.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .interfaces import (GlossDecoder, GlossOutput, SequenceFeatures, TemporalEncoder,
                         TranslationModel, VisualEncoder)


class FrameCNNEncoder(VisualEncoder):
    """Tiny per-frame CNN (frames folded into the batch). Replaceable by e.g. a
    torchvision backbone via :class:`TorchvisionFrameEncoder`."""

    modality = "rgb"

    def __init__(self, in_channels: int = 3, output_dim: int = 128, frame_chunk: int = 64):
        super().__init__()
        self.output_dim, self.frame_chunk = output_dim, frame_chunk
        c = [in_channels, 16, 32, 64]
        layers = []
        for a, b in zip(c[:-1], c[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, output_dim))

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> SequenceFeatures:
        if inputs.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(inputs.shape)}")
        B, T = inputs.shape[:2]
        mask = SequenceFeatures.mask_from_lengths(lengths, T)
        flat = inputs[mask]                                   # only real frames are encoded
        feats = [self.cnn(flat[i:i + self.frame_chunk]) for i in range(0, len(flat), self.frame_chunk)]
        out = inputs.new_zeros(B, T, self.output_dim)
        if feats:
            out[mask] = torch.cat(feats)
        return SequenceFeatures(out, lengths, mask)


class TorchvisionFrameEncoder(VisualEncoder):
    """Wraps a torchvision classification backbone (default MobileNetV2, a small CPU-friendly
    backbone). ``weights=None`` -> no download, random init."""

    modality = "rgb"

    def __init__(self, arch: str = "mobilenet_v2", weights=None, frame_chunk: int = 32):
        super().__init__()
        import torchvision
        net = getattr(torchvision.models, arch)(weights=weights)
        self.output_dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        self.net, self.frame_chunk = net, frame_chunk

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> SequenceFeatures:
        B, T = inputs.shape[:2]
        mask = SequenceFeatures.mask_from_lengths(lengths, T)
        flat = inputs[mask]
        out = inputs.new_zeros(B, T, self.output_dim)
        if len(flat):
            out[mask] = torch.cat([self.net(flat[i:i + self.frame_chunk])
                                   for i in range(0, len(flat), self.frame_chunk)])
        return SequenceFeatures(out, lengths, mask)


class PoseMLPEncoder(VisualEncoder):
    """Keypoint encoder: (B,T,K,C) -> (B,T,D). Missing keypoints should be zero +
    flagged in an extra channel by the feature extractor."""

    modality = "pose"

    def __init__(self, num_keypoints: int, channels: int, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(nn.Linear(num_keypoints * channels, 256), nn.ReLU(), nn.Linear(256, output_dim))

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> SequenceFeatures:
        B, T = inputs.shape[:2]
        mask = SequenceFeatures.mask_from_lengths(lengths, T)
        out = self.mlp(inputs.reshape(B, T, -1)) * mask[..., None]
        return SequenceFeatures(out, lengths, mask)


class TemporalConvEncoder(TemporalEncoder):
    """1D temporal convs with optional stride-2 downsampling (common in CSLR, e.g. the
    'TConv' block of VAC/CorrNet-style models)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, downsample: int = 2):
        super().__init__()
        if downsample not in (1, 2, 4):
            raise ValueError("downsample must be 1, 2 or 4")
        self.input_dim, self.output_dim, self.downsample = input_dim, hidden_dim, downsample
        blocks, dim = [], input_dim
        for _ in range(int(math.log2(downsample))):
            blocks += [nn.Conv1d(dim, hidden_dim, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True)]
            dim = hidden_dim
        blocks += [nn.Conv1d(dim, hidden_dim, 3, padding=1), nn.ReLU()]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: SequenceFeatures) -> SequenceFeatures:
        h = self.net((x.features * x.mask[..., None]).transpose(1, 2)).transpose(1, 2)
        lengths = torch.div(x.lengths + self.downsample - 1, self.downsample, rounding_mode="floor")
        lengths = lengths.clamp(min=1, max=h.shape[1])
        mask = SequenceFeatures.mask_from_lengths(lengths, h.shape[1])
        return SequenceFeatures(h * mask[..., None], lengths, mask)


class TransformerTemporalEncoder(TemporalEncoder):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, layers: int = 2,
                 max_len: int = 2048):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, d_model
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x: SequenceFeatures) -> SequenceFeatures:
        T = x.features.shape[1]
        h = self.proj(x.features) + self.pos[:, :T]
        h = self.enc(h, src_key_padding_mask=~x.mask)
        return SequenceFeatures(h * x.mask[..., None], x.lengths, x.mask)


class CTCGlossDecoder(GlossDecoder):
    def __init__(self, input_dim: int, vocab_size: int, blank_id: int = 0):
        super().__init__()
        if vocab_size < 2:
            raise ValueError("gloss vocabulary must contain blank + at least one gloss")
        self.vocab_size, self.blank_id = vocab_size, blank_id
        self.head = nn.Linear(input_dim, vocab_size)
        self.ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def forward(self, x: SequenceFeatures) -> GlossOutput:
        return GlossOutput(F.log_softmax(self.head(x.features), -1).transpose(0, 1), x.lengths, self.blank_id)

    def loss(self, out: GlossOutput, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        return self.ctc(out.log_probs, targets, out.lengths, target_lengths)

    def decode(self, out: GlossOutput) -> list[list[int]]:
        """Greedy CTC: argmax, collapse repeats, drop blanks."""
        best = out.log_probs.argmax(-1).transpose(0, 1)  # (B, T)
        res = []
        for b, n in enumerate(out.lengths.tolist()):
            seq, prev = [], None
            for t in best[b, :n].tolist():
                if t != prev and t != self.blank_id:
                    seq.append(t)
                prev = t
            res.append(seq)
        return res


class TransformerTranslationModel(TranslationModel):
    def __init__(self, memory_dim: int, vocab_size: int, pad_id: int = 0, bos_id: int = 2, eos_id: int = 3,
                 d_model: int = 128, nhead: int = 4, layers: int = 2, max_len: int = 256):
        super().__init__()
        self.vocab_size, self.pad_id, self.bos_id, self.eos_id = vocab_size, pad_id, bos_id, eos_id
        self.mem_proj = nn.Linear(memory_dim, d_model)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerDecoderLayer(d_model, nhead, 4 * d_model, batch_first=True, dropout=0.1)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, memory: SequenceFeatures, target_in: torch.Tensor) -> torch.Tensor:
        L = target_in.shape[1]
        tgt = self.emb(target_in) + self.pos[:, :L]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=target_in.device), 1)
        h = self.dec(tgt, self.mem_proj(memory.features), tgt_mask=causal,
                     tgt_key_padding_mask=target_in == self.pad_id, memory_key_padding_mask=~memory.mask)
        return self.out(h)

    @torch.no_grad()
    def generate(self, memory: SequenceFeatures, max_len: int = 50) -> list[list[int]]:
        B = memory.features.shape[0]
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=memory.features.device)
        done = torch.zeros(B, dtype=torch.bool, device=ys.device)
        for _ in range(max_len):
            nxt = self.forward(memory, ys)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == self.eos_id
            if done.all():
                break
        out = []
        for row in ys[:, 1:].tolist():
            seq = []
            for t in row:
                if t in (self.eos_id, self.pad_id):
                    break
                seq.append(t)
            out.append(seq)
        return out
