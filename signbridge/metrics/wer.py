"""Word error rate for gloss sequences (CSLR).

WER = (S + D + I) / N, where N is the number of reference tokens and S/D/I come from a
weighted minimum-edit alignment (sclite weights by default). Corpus WER sums edits and
reference lengths over all sentences; it is **not** the mean of per-sentence WERs.

For PHOENIX, apply the official gloss normalisation first (``signbridge.metrics.phoenix``).
That combination reproduced the official sclite output on the release's example hypotheses.
Still cite the official script for published numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        """Errors / reference length. Raises on an empty reference (WER is undefined)."""
        if self.reference_length == 0:
            raise ValueError("WER is undefined for an empty reference")
        return self.errors / self.reference_length

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )


def _tokens(x: str | Sequence[str]) -> list[str]:
    if isinstance(x, str):
        return x.split()
    return list(x)


SCLITE_WEIGHTS = (4, 3, 3)  # NIST sclite defaults: substitution, deletion, insertion


def edit_counts(
    reference: str | Sequence[str],
    hypothesis: str | Sequence[str],
    weights: tuple[int, int, int] = SCLITE_WEIGHTS,
) -> EditCounts:
    """Align ``hypothesis`` to ``reference`` and count S, D and I.

    The alignment minimises ``ws*S + wd*D + wi*I``. The default uses sclite's weights
    (4, 3, 3), which reproduced the official PHOENIX-2014-T sclite output exactly, including
    the S/D/I split (``scripts/check_phoenix_wer.py``). ``(1, 1, 1)`` gives plain Levenshtein.
    Strings are split on whitespace; sequences are used as given.
    """
    ws, wd, wi = weights
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    n, m = len(ref), len(hyp)
    # cell = (weighted cost, subs, dels, ins) for ref[:i] vs hyp[:j]
    prev = [(j * wi, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i * wd, 0, i, 0)] + [None] * m  # type: ignore[list-item]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                diag = prev[j - 1]
            else:
                t, s, d, ins = prev[j - 1]
                diag = (t + ws, s + 1, d, ins)
            t, s, d, ins = prev[j]
            dele = (t + wd, s, d + 1, ins)
            t, s, d, ins = cur[j - 1]
            inse = (t + wi, s, d, ins + 1)
            cur[j] = min(diag, dele, inse, key=lambda c: c[0])
        prev = cur
    _, s, d, ins = prev[m]
    return EditCounts(s, d, ins, n)


def corpus_wer(references: Iterable[str | Sequence[str]], hypotheses: Iterable[str | Sequence[str]]) -> EditCounts:
    """Sum edit counts over aligned (reference, hypothesis) pairs. Lengths must match."""
    refs = list(references)
    hyps = list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    total = EditCounts()
    for r, h in zip(refs, hyps):
        total = total + edit_counts(r, h)
    return total
