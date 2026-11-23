"""Evaluate a trained greeting-sign checkpoint on the held-out ASL Citizen test signers.

Reports sign-level accuracy (29 classes) and word-level accuracy, where lexical variants of
the same English word (FINE1/FINE2, HOW1/HOW2, WHAT1/WHAT2, DEAF1/DEAF2) count as correct
for each other. The demo's phrase table maps them to the same word, so word-level is what a
viewer sees. Also reports the effect of the demo's confidence threshold.

    python -X utf8 scripts/eval_sign_classifier.py [--ckpt ../models/signs/asl_greetings.pt]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.signs.features import RawSequence, sequence_features  # noqa: E402
from signbridge.signs.live import LiveConfig  # noqa: E402
from signbridge.signs.model import load_checkpoint  # noqa: E402
from signbridge.signs.phrases import base_gloss  # noqa: E402

DATA = REPO.parent / "datasets" / "asl_citizen"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO.parent / "models" / "signs" / "asl_greetings.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=LiveConfig.min_confidence)
    args = ap.parse_args()
    model, classes, t, meta = load_checkpoint(args.ckpt)
    rows = [r for r in csv.DictReader(open(DATA / "landmarks_index.csv", encoding="utf-8"))
            if r["split"] == args.split]
    X = np.stack([sequence_features(RawSequence.load(DATA / "landmarks" / f"{r['video']}.npz"), t) for r in rows])
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(X)), 1).numpy()
    pred = [classes[i] for i in p.argmax(1)]
    conf = p.max(1)
    gold = [r["gloss"] for r in rows]
    sign_ok = np.array([a == b for a, b in zip(pred, gold)])
    word_ok = np.array([base_gloss(a) == base_gloss(b) for a, b in zip(pred, gold)])
    acc = conf >= args.threshold
    signers = sorted({r["signer"] for r in rows})
    per_signer = {s: float(word_ok[[r["signer"] == s for r in rows]].mean()) for s in signers}
    word_conf = Counter(f"{base_gloss(g)}->{base_gloss(q)}" for g, q in zip(gold, pred) if base_gloss(g) != base_gloss(q))
    res = {
        "ckpt": str(args.ckpt.name), "split": args.split, "n": len(rows), "signers": len(signers),
        "sign_top1": float(sign_ok.mean()), "word_top1": float(word_ok.mean()),
        "threshold": args.threshold,
        "accepted_rate": float(acc.mean()),
        "word_top1_when_accepted": float(word_ok[acc].mean()) if acc.any() else None,
        "wrong_words_shown_rate": float((acc & ~word_ok).mean()),
        "per_signer_word_top1": {k: round(v, 3) for k, v in per_signer.items()},
        "word_confusions": word_conf.most_common(10),
    }
    print(json.dumps(res, indent=1))
    out = REPO / "outputs" / "signs" / f"eval_{args.ckpt.stem}_{args.split}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
