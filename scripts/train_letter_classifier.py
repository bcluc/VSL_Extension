"""Train the static VSL letter classifier on takes recorded with scripts/record_letters.py.

    python -X utf8 scripts/train_letter_classifier.py
    python -X utf8 scripts/train_letter_classifier.py --test-signer s02   # hold out a whole signer

Split (whole takes, never frames of the same take on both sides):
* default: per signer and letter, the newest take -> test, the next newest -> val, the rest ->
  train (needs >= 3 takes per letter). With a single signer this measures how well the model
  recognises **that signer** later on, not a new person: it overestimates accuracy for others.
* ``--test-signer``: every take of that signer -> test (a real unseen-signer check).

Writes ../models/letters/vsl_letters.pt (outside Git) and outputs/letters/train_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.letters.features import FEATURE_DIM, augment, letter_features  # noqa: E402
from signbridge.letters.labels import DEFAULT_LABELS  # noqa: E402
from signbridge.letters.model import LetterMLP, save_letter_model  # noqa: E402
from signbridge.letters.recordings import Take, load_takes  # noqa: E402

DEFAULT_ROOT = REPO.parent / "datasets" / "vsl_letters"
DEFAULT_OUT = REPO.parent / "models" / "letters" / "vsl_letters.pt"
REPORT = REPO / "outputs" / "letters" / "train_report.json"


def split_takes(takes: list[Take], test_signer: str | None) -> dict[str, list[Take]]:
    out: dict[str, list[Take]] = {"train": [], "val": [], "test": []}
    groups: dict[tuple[str, str], list[Take]] = defaultdict(list)
    for t in takes:
        if test_signer and t.signer == test_signer:
            out["test"].append(t)
        else:
            groups[(t.signer, t.label)].append(t)
    for (signer, label), g in sorted(groups.items()):
        g = sorted(g, key=lambda t: t.path.name)
        need = 2 if test_signer else 3
        if len(g) < need:
            raise SystemExit(f"{signer}/{label}: {len(g)} takes, need at least {need}. Record more.")
        if test_signer:
            out["val"].append(g[-1])
            out["train"].extend(g[:-1])
        else:
            out["test"].append(g[-1])
            out["val"].append(g[-2])
            out["train"].extend(g[:-2])
    return out


def arrays(takes: list[Take], cls: dict[str, int]):
    lms = np.concatenate([t.landmarks for t in takes])
    left = np.concatenate([t.is_left for t in takes])
    asp = np.concatenate([np.full(len(t), t.aspect, np.float32) for t in takes])
    y = np.concatenate([np.full(len(t), cls[t.label]) for t in takes])
    take_id = np.concatenate([np.full(len(t), i) for i, t in enumerate(takes)])
    return lms, left, asp, y, take_id


@torch.no_grad()
def evaluate(model: LetterMLP, takes: list[Take], cls: dict[str, int], labels: list[str]) -> dict:
    if not takes:
        return {}
    lms, left, asp, y, tid = arrays(takes, cls)
    probs = torch.softmax(model(torch.from_numpy(letter_features(lms, left, asp))), -1).numpy()
    pred = probs.argmax(1)
    take_pred = np.array([probs[tid == i].mean(0).argmax() for i in range(len(takes))])
    take_true = np.array([cls[t.label] for t in takes])
    per_label = {labels[c]: round(float((pred[y == c] == c).mean()), 4) for c in np.unique(y)}
    conf = Counter((labels[a], labels[b]) for a, b in zip(y, pred) if a != b)
    return {"frames": int(len(y)), "takes": len(takes),
            "frame_top1": round(float((pred == y).mean()), 4),
            "frame_top3": round(float((np.argsort(-probs, 1)[:, :3] == y[:, None]).any(1).mean()), 4),
            "take_top1": round(float((take_pred == take_true).mean()), 4),
            "per_label_frame_top1": per_label,
            "top_confusions": [[a, b, n] for (a, b), n in conf.most_common(10)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--labels", nargs="+", default=[k for k, _ in DEFAULT_LABELS])
    ap.add_argument("--test-signer")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    takes = load_takes(args.root, args.labels)
    if not takes:
        raise SystemExit(f"No takes under {args.root}. Record with scripts/record_letters.py first.")
    labels = [lb for lb in args.labels if any(t.label == lb for t in takes)]
    missing = sorted(set(args.labels) - set(labels))
    if missing:
        print(f"[!] no takes for {missing}: they are left out of the model")
    cls = {lb: i for i, lb in enumerate(labels)}
    split = split_takes(takes, args.test_signer)
    for k, v in split.items():
        print(f"{k}: {len(v)} takes, {sum(len(t) for t in v)} frames")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    lms, left, asp, y, _ = arrays(split["train"], cls)
    model = LetterMLP(len(labels), FEATURE_DIM, args.hidden)
    model.set_normalisation(letter_features(lms, left, asp))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    yt = torch.from_numpy(y).long()
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        x = torch.from_numpy(letter_features(augment(lms, rng), left, asp))
        perm = torch.randperm(len(x))
        for b in range(0, len(x), 256):
            idx = perm[b:b + 256]
            opt.zero_grad()
            lossf(model(x[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        v = evaluate(model, split["val"], cls, labels)["frame_top1"]
        if v > best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  val frame top-1 {v:.3f}")
    model.load_state_dict(best_state)
    model.eval()

    signers = sorted({t.signer for t in takes})
    report = {"labels": labels, "signers": signers, "test_signer": args.test_signer,
              "split": "held-out signer" if args.test_signer else "newest take per signer and letter",
              "caveat": None if args.test_signer else
              "same signer(s) in train and test: does not measure accuracy for a new person",
              "val": evaluate(model, split["val"], cls, labels),
              "test": evaluate(model, split["test"], cls, labels)}
    save_letter_model(args.out, model, labels, args.hidden,
                      {"signers": signers, "test_signer": args.test_signer,
                       "test_frame_top1": report["test"].get("frame_top1"), "seed": args.seed})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    t = report["test"]
    print(f"test: frame top-1 {t['frame_top1']:.3f}, top-3 {t['frame_top3']:.3f}, "
          f"take top-1 {t['take_top1']:.3f} ({t['takes']} takes)")
    if report["caveat"]:
        print(f"[!] {report['caveat']}")
    print(f"model -> {args.out}\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
