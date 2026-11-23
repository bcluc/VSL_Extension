"""PHOENIX-2014-T gloss normalisation, ported from the official evaluation script.

Source: ``evaluation/sign-recognition/evaluatePHOENIX-2014-T-signrecognition.sh`` in the
PHOENIX-2014-T release v3. The shell pipeline (grep/sed/awk/perl) normalises the hypothesis
CTM and the reference STM differently before scoring with NIST sclite. This module reproduces
both pipelines rule by rule, in the same order, so WER can be computed without sclite.

Agreement with the official scorer is checked in ``scripts/check_phoenix_wer.py`` against the
sclite outputs shipped with the release. Use the official script for numbers you publish.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from .wer import EditCounts, edit_counts

_A = re.ASCII
_HYP_DROP = ("__LEFTHAND__", "__EPENTHESIS__", "__EMOTION__")
_HYP_META = re.compile(r"^__[^_ ]*__$", _A)
_HYP_RAUM = re.compile(r"\b([A-Z][A-Z]*)RAUM$", _A)

# Reference rules, applied in order to a whole STM-like line (header + glosses).
_REF_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"__LEFTHAND__ "), ""), (re.compile(r" __LEFTHAND__"), ""),
    (re.compile(r" __EPENTHESIS__"), ""), (re.compile(r"__EPENTHESIS__ "), ""),
    (re.compile(r" __EMOTION__"), ""), (re.compile(r"__EMOTION__ "), ""),
    (re.compile(r"\b__[^_ ]*__\b", _A), ""),
    (re.compile(r"\bloc-([^ ]*)\b", _A), r"\1"),
    (re.compile(r"\bcl-([^ ]*)\b", _A), r"\1"),
    (re.compile(r"\b([^ ]*)-PLUSPLUS\b", _A), r"\1"),
    (re.compile(r"\b([A-Z][A-Z]*)RAUM\b", _A), r"\1"),
    (re.compile(r"WIE AUSSEHEN"), "WIE-AUSSEHEN"),
    (re.compile(r"^([A-Z]) ([A-Z][+ ])"), r"\1+\2"),
    (re.compile(r"[ +]([A-Z]) ([A-Z]) "), r" \1+\2 "),
    (re.compile(r"([ +][A-Z]) ([A-Z][ +])"), r"\1+\2"),
    (re.compile(r"([ +][A-Z]) ([A-Z][ +])"), r"\1+\2"),
    (re.compile(r"([ +][A-Z]) ([A-Z][ +])"), r"\1+\2"),
    (re.compile(r"([ +]SCH) ([A-Z][ +])"), r"\1+\2"),
    (re.compile(r"([ +]NN) ([A-Z][ +])"), r"\1+\2"),
    (re.compile(r"([ +][A-Z]) (NN[ +])"), r"\1+\2"),
    (re.compile(r"([ +][A-Z]) ([A-Z])$"), r"\1+\2"),
]
_REF_REPEAT = re.compile(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", _A)
_STM_HEADER = "SAMPLE 1 SIGNER 0.0 1.79769e+308 "  # neutral header; never matched by the rules


def normalize_hypothesis(glosses: list[str]) -> list[str]:
    """Hypothesis pipeline (CTM side): drop meta tokens, strip prefixes/suffixes, collapse repeats."""
    out: list[str] = []
    for w in glosses:
        if any(t in w for t in _HYP_DROP) or _HYP_META.match(w):
            continue
        if w.startswith("loc-"):
            w = w[4:]
        if w.endswith("-PLUSPLUS"):
            w = w[: -len("-PLUSPLUS")]
        if w.startswith("cl-"):
            w = w[3:]
        w = _HYP_RAUM.sub(r"\1", w).rstrip()
        if not w:
            continue
        if out and out[-1] == w:  # awk: drop a token equal to the previous one
            continue
        out.append(w)
    return out


def _normalize_ref_line(line: str) -> str:
    for pat, rep in _REF_SUBS:
        line = pat.sub(rep, line)
    for _ in range(4):  # the script runs the perl repetition filter four times
        line = _REF_REPEAT.sub(r"\1", line)
    return line


def normalize_reference(glosses: list[str] | str) -> list[str]:
    """Reference pipeline (STM side), including compounding of spelled letters (A S -> A+S)."""
    text = glosses if isinstance(glosses, str) else " ".join(glosses)
    line = _normalize_ref_line(_STM_HEADER + text)
    return line.split()[5:]


def read_stm(path: str | Path) -> "OrderedDict[str, list[str]]":
    """Read an STM file: ``id channel speaker start end gloss...`` -> {id: raw glosses}."""
    refs: OrderedDict[str, list[str]] = OrderedDict()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            refs[parts[0]] = parts[5:]
    return refs


def read_ctm(path: str | Path) -> "OrderedDict[str, list[str]]":
    """Read a CTM file: ``id channel start duration gloss`` per line -> {id: glosses}."""
    hyps: OrderedDict[str, list[str]] = OrderedDict()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            hyps.setdefault(parts[0], []).append(parts[4])
    return hyps


def phoenix_wer(references: dict[str, list[str]], hypotheses: dict[str, list[str]]) -> EditCounts:
    """Corpus WER with official normalisation. Missing hypotheses count as empty output."""
    total = EditCounts()
    for sid, ref in references.items():
        hyp = normalize_hypothesis(hypotheses.get(sid, []))
        total = total + edit_counts(normalize_reference(ref), hyp)
    return total
