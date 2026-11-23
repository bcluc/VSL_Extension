"""Train the greeting-sign classifier on ASL Citizen landmarks; report held-out-signer accuracy.

Splits are ASL Citizen's official ones, which are signer-independent: the test signers never
appear in training. The model is selected on ``val`` only; ``test`` is evaluated once at the end.

    python -X utf8 scripts/train_sign_classifier.py [--epochs 150] [--seed 0]

Writes ``../models/signs/asl_greetings.pt`` (outside Git) and ``outputs/signs/results_seed<k>.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.signs.features import RawSequence, active_span, mirror, sequence_features  # noqa: E402
from signbridge.signs.model import SignClassifier, save_checkpoint  # noqa: E402

DATA = REPO.parent / "datasets" / "asl_citizen"
CKPT = REPO.parent / "models" / "signs" / "asl_greetings.pt"
OUT = REPO / "outputs" / "signs"


def augment(seq: RawSequence, rng: np.random.Generator, t: int) -> np.ndarray:
    if rng.random() < 0.5:
        seq = mirror(seq)
    pose, hands = seq.pose.copy(), seq.hands.copy()
    # small similarity transform in image space around the frame centre
    ang = np.deg2rad(rng.uniform(-10, 10))
    sc = rng.uniform(0.9, 1.1)
    sh = rng.uniform(-0.05, 0.05, size=2)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * sc
    for arr, mask in ((pose[..., :2], seq.pose_present[:, None]), (hands[..., :2], seq.hand_present[..., None])):
        moved = (arr - 0.5) @ rot.T + 0.5 + sh
        arr[...] = np.where(mask[..., None], moved, arr)
    aug = RawSequence(pose, hands, seq.hand_present, seq.pose_present)
    # temporal jitter of the active span
    s, e = active_span(seq.hand_present)
    n = e - s
    if n > 8:
        ds = int(rng.integers(0, max(1, n // 8)))
        de = int(rng.integers(0, max(1, n // 8)))
        s, e = s + ds, e - de
    feats = sequence_features(aug, t, (s, e))
    # random frame dropout (replace by the previous frame)
    drop = rng.random(t) < 0.1
    drop[0] = False
    for i in np.flatnonzero(drop):
        feats[i] = feats[i - 1]
    return feats


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
    prob = torch.softmax(logits, 1).numpy()
    top = np.argsort(-prob, 1)
    return {"top1": float((top[:, 0] == y).mean()), "top3": float((top[:, :3] == y[:, None]).any(1).mean()),
            "pred": top[:, 0], "conf": prob.max(1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)

    rows = list(csv.DictReader(open(DATA / "landmarks_index.csv", encoding="utf-8")))
    rows = [r for r in rows if int(r["frames"]) > 0]
    classes = sorted({r["gloss"] for r in rows})
    cid = {c: i for i, c in enumerate(classes)}
    seqs = {r["video"]: RawSequence.load(DATA / "landmarks" / f"{r['video']}.npz") for r in rows}
    split = defaultdict(list)
    for r in rows:
        split[r["split"]].append(r)
    signers = {k: sorted({r["signer"] for r in v}) for k, v in split.items()}
    assert not (set(signers["train"]) & set(signers["test"])), "train/test signers overlap"

    def fixed(rs):
        return (np.stack([sequence_features(seqs[r["video"]], args.t) for r in rs]),
                np.array([cid[r["gloss"]] for r in rs]))

    Xva, yva = fixed(split["val"])
    Xte, yte = fixed(split["test"])
    tr = split["train"]
    ytr = np.array([cid[r["gloss"]] for r in tr])
    rng = np.random.default_rng(args.seed)

    model = SignClassifier(Xva.shape[2], len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * ((len(tr) + 31) // 32))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, best_ep = -1.0, None, -1
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        for b in range(0, len(order), 32):
            idx = order[b:b + 32]
            xb = torch.from_numpy(np.stack([augment(seqs[tr[i]["video"]], rng, args.t) for i in idx]))
            loss = lossf(model(xb), torch.from_numpy(ytr[idx]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        va = evaluate(model, Xva, yva)
        if va["top1"] > best:
            best, best_ep = va["top1"], ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} loss {loss.item():.3f} val top1 {va['top1']:.3f} top3 {va['top3']:.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)

    model.load_state_dict(best_state)
    va = evaluate(model, Xva, yva)
    te = evaluate(model, Xte, yte)
    per_class = {}
    conf = Counter()
    for c in classes:
        m = yte == cid[c]
        per_class[c] = {"n": int(m.sum()), "acc": float((te["pred"][m] == cid[c]).mean()) if m.any() else None}
    for yt, yp in zip(yte, te["pred"]):
        if yt != yp:
            conf[f"{classes[yt]}->{classes[yp]}"] += 1
    res = {
        "seed": args.seed, "epochs": args.epochs, "best_epoch": best_ep, "t": args.t,
        "classes": classes, "n": {k: len(v) for k, v in split.items()},
        "signers": {k: len(v) for k, v in signers.items()},
        "val": {"top1": va["top1"], "top3": va["top3"]},
        "test": {"top1": te["top1"], "top3": te["top3"]},
        "test_per_class": per_class, "test_top_confusions": conf.most_common(15),
        "train_seconds": round(time.time() - t0),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_seed{args.seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    save_checkpoint(CKPT if args.seed == 0 else CKPT.with_name(f"asl_greetings_seed{args.seed}.pt"),
                    model, classes, args.t, {"val_top1": va["top1"], "test_top1": te["top1"],
                                             "data": "ASL Citizen (Microsoft Research License, non-commercial)"})
    print(json.dumps({k: res[k] for k in ("n", "signers", "val", "test", "best_epoch")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
