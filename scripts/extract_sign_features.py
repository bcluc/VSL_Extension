"""Extract pose + hand landmarks from the ASL Citizen subset (parallel, CPU).

    python -I -X utf8 scripts/extract_sign_features.py [--workers 16]

Reads ``../datasets/asl_citizen/subset.csv`` and ``videos/``; writes one ``.npz`` per clip to
``../datasets/asl_citizen/landmarks/`` plus ``landmarks_index.csv`` (frames, fps, detection rates).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA = REPO.parent / "datasets" / "asl_citizen"
MODELS = REPO.parent / "models" / "mediapipe"

_ext = None


def _init():
    global _ext
    from signbridge.signs.features import LandmarkExtractor
    _ext = LandmarkExtractor(MODELS / "pose_landmarker_lite.task", MODELS / "hand_landmarker.task")


def _work(video: str, out: str) -> dict:
    seq, fps = _ext.video(video)
    seq.save(Path(out), fps=fps)
    n = len(seq)
    return {"frames": n, "fps": round(fps, 3),
            "pose_rate": round(float(seq.pose_present.mean()), 3) if n else 0.0,
            "hand_rate": round(float(seq.hand_present.any(axis=1).mean()), 3) if n else 0.0}


def _info_from_npz(path: Path) -> dict:
    import numpy as np
    z = np.load(path)
    n = len(z["pose"])
    return {"frames": n, "fps": round(float(z["fps"]), 3),
            "pose_rate": round(float(z["pose_present"].mean()), 3) if n else 0.0,
            "hand_rate": round(float(z["hand_present"].any(axis=1).mean()), 3) if n else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6, help="each worker holds two MediaPipe graphs (~1 GB)")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(DATA / "subset.csv", encoding="utf-8")))
    out_dir = DATA / "landmarks"
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    todo = []
    for r in rows:
        v = DATA / "videos" / Path(r["Video file"]).name
        if not v.exists():
            print(f"missing video {v.name}")
        elif not (out_dir / (v.stem + ".npz")).exists():  # resumable
            todo.append(v)
    workers, breaks = args.workers, 0
    while todo:
        # A crashing worker breaks the whole pool: rebuild it with the remaining clips, and after
        # repeated breaks isolate clips one per pool so a single bad clip is skipped, not fatal.
        failed_now = []
        try:
            with ProcessPoolExecutor(workers, initializer=_init) as ex:
                futs = {ex.submit(_work, str(v), str(out_dir / (v.stem + ".npz"))): v for v in todo}
                for k, fut in enumerate(as_completed(futs), 1):
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001
                        failed_now.append((futs[fut], e))
                    if k % 100 == 0:
                        print(f"{k}/{len(futs)} clips this pass, {time.time() - t0:.0f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"pool error: {e}")
        remaining = [v for v in todo if not (out_dir / (v.stem + ".npz")).exists()]
        if len(remaining) == len(todo) and workers == 1 and len(todo) == 1:
            print(f"SKIP (crashes the extractor): {todo[0].name}: {failed_now[:1]}")
            remaining = []
        elif remaining:
            breaks += 1
            print(f"{len(remaining)} clips left after pass {breaks} (workers={workers})", flush=True)
            if breaks >= 3:
                workers = 1
                # one clip per pool from now on
                for v in remaining:
                    try:
                        with ProcessPoolExecutor(1, initializer=_init) as ex:
                            ex.submit(_work, str(v), str(out_dir / (v.stem + ".npz"))).result()
                    except Exception as e:  # noqa: BLE001
                        print(f"SKIP {v.name}: {e}")
                remaining = []
        todo = remaining
    results = []
    for r in rows:
        stem = Path(r["Video file"]).stem
        p = out_dir / f"{stem}.npz"
        if p.exists():
            results.append({"video": stem, "gloss": r["Gloss"], "signer": r["Participant ID"],
                            "split": r["split"], **_info_from_npz(p)})
    results.sort(key=lambda x: x["video"])
    with open(DATA / "landmarks_index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"done: {len(results)} clips in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
