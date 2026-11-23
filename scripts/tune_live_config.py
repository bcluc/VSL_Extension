"""Replay the live state machine on cached per-frame landmarks to compare ``LiveConfig`` settings.

Landmarks of each conversation video (from ``make_demo_test_video.py``) are extracted once,
exactly as the live demo does it (MediaPipe VIDEO mode, frame timestamps from the video fps),
and cached in ``outputs/signs/landmark_cache/``. Each setting then re-runs only
``SignSession``, which is fast. Score = word error rate of accepted signs vs. the script
(lexical variants merged) and sentences reproduced exactly, as in ``score_demo_test.py``.

Tune on ``--split val`` only. Run the chosen setting once on ``--split test`` for the report.

    python -X utf8 scripts/tune_live_config.py --split val --grid
    python -X utf8 scripts/tune_live_config.py --split test --set active_window=3 active_min=2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.metrics import EditCounts, edit_counts  # noqa: E402
from signbridge.signs.live import LiveConfig, SentenceEvent, SignEvent, SignSession  # noqa: E402
from signbridge.signs.model import load_ensemble  # noqa: E402
from signbridge.signs.phrases import base_gloss, compose  # noqa: E402

DATA = REPO.parent / "datasets" / "asl_citizen"
MODELS = REPO.parent / "models"
CACHE = REPO / "outputs" / "signs" / "landmark_cache"
DEFAULT_CKPTS = [MODELS / "signs" / n for n in ("asl_greetings.pt", "asl_greetings_seed1.pt", "asl_greetings_seed2.pt")]
GRID = {"active_window": [(1, 1), (3, 2), (5, 3)], "end_gap_s": [0.3, 0.4, 0.6]}


def cached_landmarks(video: Path, mode: str) -> dict:
    import cv2

    from signbridge.signs.features import LandmarkExtractor
    out = CACHE / f"{video.parent.name}_{video.stem}_{mode}.npz"
    if out.exists():
        return dict(np.load(out))
    ext = LandmarkExtractor(MODELS / "mediapipe" / "pose_landmarker_lite.task",
                            MODELS / "mediapipe" / "hand_landmarker.task", mode=mode, parallel=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    P, H, PR, OK = [], [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pose, hands, present, pose_ok = ext.frame(frame, int(len(P) / fps * 1000))
        P.append(pose), H.append(hands), PR.append(present), OK.append(pose_ok)
    cap.release()
    ext.close()
    d = {"pose": np.stack(P), "hands": np.stack(H), "present": np.stack(PR), "pose_ok": np.array(OK), "fps": fps}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    return d


def replay(lm: dict, model, classes, t, cfg: LiveConfig) -> tuple[list[str], list[str]]:
    s = SignSession(model, classes, t, cfg)
    words, sents = [], []
    n = len(lm["pose"])
    fps = float(lm["fps"])

    def take(evs):
        for ev in evs:
            if isinstance(ev, SignEvent) and ev.gloss and ev.accepted:
                words.append(base_gloss(ev.gloss))
            elif isinstance(ev, SentenceEvent):
                sents.append(ev.composed.english)
    for i in range(n):
        take(s.push(i / fps, lm["pose"][i], lm["hands"][i], lm["present"][i], bool(lm["pose_ok"][i])))
    take(s.flush(n / fps))
    return words, sents


def lcs(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def score(videos, model, classes, t, cfg) -> dict:
    total, exact, n_sent = EditCounts(), 0, 0
    for gt, lm in videos:
        words, sents = replay(lm, model, classes, t, cfg)
        total = total + edit_counts([base_gloss(x) for s in gt["sentences"] for x in s], words)
        ref = [compose(s).english for s in gt["sentences"]]
        exact += lcs(ref, sents)
        n_sent += len(ref)
    return {"WER": round(100 * total.wer, 1), "S/D/I": [total.substitutions, total.deletions, total.insertions],
            "sentences_exact": f"{exact}/{n_sent}", "_exact": exact}


def parse_set(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = type(getattr(LiveConfig, k))(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--ckpt", type=Path, nargs="+", default=DEFAULT_CKPTS)
    ap.add_argument("--mp-mode", choices=["image", "video"], default="video")
    ap.add_argument("--grid", action="store_true", help="try the GRID settings")
    ap.add_argument("--set", nargs="*", default=[], help="LiveConfig overrides, e.g. active_window=3")
    args = ap.parse_args()
    folder = DATA / f"demo_{args.split}"
    gts = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
    videos = [(g, cached_landmarks(folder / g["video"], args.mp_mode)) for g in gts]
    model, classes, t, _ = load_ensemble(args.ckpt)
    base = replace(LiveConfig(), **parse_set(args.set))
    configs = []
    if args.grid:
        for (w, m), gap in itertools.product(GRID["active_window"], GRID["end_gap_s"]):
            configs.append(replace(base, active_window=w, active_min=m, end_gap_s=gap))
    else:
        configs.append(base)
    rows = []
    for cfg in configs:
        r = score(videos, model, classes, t, cfg)
        r = {"active": f"{cfg.active_min}/{cfg.active_window}", "end_gap_s": cfg.end_gap_s,
             "min_conf": cfg.min_confidence, **r}
        rows.append(r)
        print({k: v for k, v in r.items() if not k.startswith("_")}, flush=True)
    out = REPO / "outputs" / "signs" / f"tune_live_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "ckpts": [p.name for p in args.ckpt], "mp_mode": args.mp_mode,
                               "videos": len(videos), "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
