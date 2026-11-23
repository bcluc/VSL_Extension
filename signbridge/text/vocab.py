"""Vocabularies for gloss sequences (CTC) and target text (translation).

Built only from real annotations. An empty gloss vocabulary is an explicit error: it
means the dataset cannot supervise a gloss decoder.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

_TOKEN_RX = re.compile(r"\w+|[^\w\s]", re.UNICODE)  # unicode \w keeps Vietnamese diacritics


def tokenize_text(text: str, lowercase: bool = True) -> list[str]:
    return _TOKEN_RX.findall(text.lower() if lowercase else text)


class Vocabulary:
    def __init__(self, tokens: Sequence[str], specials: Sequence[str]):
        self.specials = list(specials)
        self.itos = list(specials) + [t for t in tokens if t not in specials]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unk = self.stoi.get("<unk>")
        out = []
        for t in tokens:
            if t in self.stoi:
                out.append(self.stoi[t])
            elif unk is not None:
                out.append(unk)
            else:
                raise KeyError(f"token {t!r} not in vocabulary and no <unk>")
        return out

    def decode(self, ids: Iterable[int], skip_specials: bool = True) -> list[str]:
        return [self.itos[i] for i in ids if not (skip_specials and self.itos[i] in self.specials)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"specials": self.specials, "itos": self.itos},
                                         ensure_ascii=False, indent=1), encoding="utf-8")


class GlossVocabulary(Vocabulary):
    """CTC convention: index 0 is the blank."""

    BLANK, PAD, UNK = "<blank>", "<pad>", "<unk>"

    def __init__(self, glosses: Sequence[str]):
        super().__init__(glosses, [self.BLANK, self.PAD, self.UNK])
        self.blank_id, self.pad_id = 0, 1

    @classmethod
    def from_sequences(cls, seqs: Iterable[Sequence[str] | None], min_freq: int = 1) -> "GlossVocabulary":
        c = Counter(g for s in seqs if s for g in s)
        tokens = sorted(t for t, n in c.items() if n >= min_freq)
        if not tokens:
            raise ValueError("no gloss annotations found: a gloss (CTC) decoder cannot be "
                             "configured or trained on this data")
        return cls(tokens)


class TextVocabulary(Vocabulary):
    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

    def __init__(self, tokens: Sequence[str], lowercase: bool = True):
        super().__init__(tokens, [self.PAD, self.UNK, self.BOS, self.EOS])
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3
        self.lowercase = lowercase

    @classmethod
    def from_texts(cls, texts: Iterable[str | None], min_freq: int = 1,
                   lowercase: bool = True) -> "TextVocabulary":
        c = Counter(t for s in texts if s for t in tokenize_text(s, lowercase))
        tokens = sorted(t for t, n in c.items() if n >= min_freq)
        if not tokens:
            raise ValueError("no target text found: a translation model cannot be trained on this data")
        return cls(tokens, lowercase)

    def encode_text(self, text: str, add_bos_eos: bool = True) -> list[int]:
        ids = self.encode(tokenize_text(text, self.lowercase))
        return [self.bos_id] + ids + [self.eos_id] if add_bos_eos else ids
