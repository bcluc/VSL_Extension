"""Check signbridge's PHOENIX WER against the official sclite outputs shipped with the release.

The release contains, for dev and test:
* ``example-hypothesis-<split>.ctm``      raw example recogniser output,
* ``out.example-hypothesis-<split>.ctm``  the same after the official hypothesis normalisation,
* ``out.example-hypothesis-<split>.ctm.sys`` sclite's summary (Sub/Del/Ins/Err %).

This script (1) compares our hypothesis normalisation token-by-token with ``out.*.ctm`` and
(2) recomputes WER and compares it with the sclite ``Sum/Avg`` row. Nothing is executed from
the dataset; its files are only read.

    python -I -X utf8 scripts/check_phoenix_wer.py [--root ../datasets/phoenix2014T/...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signbridge.metrics.phoenix import normalize_hypothesis, phoenix_wer, read_ctm, read_stm  # noqa: E402

DEFAULT_ROOT = (Path(__file__).resolve().parents[2] / "datasets" / "phoenix2014T"
                / "PHOENIX-2014-T-release-v3" / "PHOENIX-2014-T")


def sclite_sum(sys_file: Path) -> dict:
    for line in sys_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Sum/Avg" in line:
            nums = [float(x) for x in re.findall(r"[\d.]+", line)]
            # | Sum/Avg | #Snt #Wrd | Corr Sub Del Ins Err S.Err |
            keys = ["sentences", "words", "corr", "sub", "del", "ins", "err", "serr"]
            return dict(zip(keys, nums))
    raise ValueError(f"no Sum/Avg row in {sys_file}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    ev = args.root / "evaluation" / "sign-recognition"
    report = {}
    ok = True
    for split in ("dev", "test"):
        refs = read_stm(ev / f"PHOENIX-2014-T-groundtruth-{split}.stm")
        raw = read_ctm(ev / f"example-hypothesis-{split}.ctm")
        official_norm = read_ctm(ev / f"out.example-hypothesis-{split}.ctm")
        ours = {k: normalize_hypothesis(v) for k, v in raw.items()}
        mismatched = [k for k in official_norm
                      if [w for w in official_norm[k] if w != "[EMPTY]"] != ours.get(k, [])]
        counts = phoenix_wer(refs, raw)
        sc = sclite_sum(ev / f"out.example-hypothesis-{split}.ctm.sys")
        ours_pct = {
            "words": counts.reference_length,
            "sub": round(100 * counts.substitutions / counts.reference_length, 1),
            "del": round(100 * counts.deletions / counts.reference_length, 1),
            "ins": round(100 * counts.insertions / counts.reference_length, 1),
            "err": round(100 * counts.wer, 1),
            "err_exact": 100 * counts.wer,
        }
        report[split] = {
            "sentences": len(refs),
            "hyp_normalisation_mismatches": len(mismatched),
            "mismatch_examples": mismatched[:5],
            "ours": ours_pct,
            "sclite": sc,
        }
        ok &= (not mismatched and ours_pct["words"] == sc["words"]
               and all(abs(ours_pct[k] - sc[k]) <= 0.05 for k in ("sub", "del", "ins", "err")))
    print(json.dumps(report, indent=1))
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
