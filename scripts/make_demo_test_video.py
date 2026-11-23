"""Build end-to-end test videos for the live demo from held-out ASL Citizen *test* signers.

For each test signer that has every needed sign, the clips of a scripted conversation are
concatenated (same signer), with a 3 s still frame (no hands) between sentences. The result
plus a ground-truth JSON lets ``live_sign_demo.py --video ... --headless`` be scored on
signers the model never saw. Videos stay in ``../datasets/asl_citizen/demo_test/`` because
the ASL Citizen license forbids redistribution.

    python -X utf8 scripts/make_demo_test_video.py [--max-signers 4] [--split val]

``--split val`` builds the same conversations from the *validation* signers (output in
``demo_val/``), so live-demo settings can be tuned without touching the test signers.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "datasets" / "asl_citizen"
SCRIPT = [["HELLO"], ["NICE", "MEET", "YOU"], ["HOW1", "YOU"], ["ME", "FINE1"],
          ["YOUR", "NAME", "WHAT1"], ["ME", "DEAF1"], ["THANKYOU"], ["SEE", "YOU", "LATER"], ["BYE"]]
W, H, FPS = 640, 480, 30.0


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f, (W, H)))
    cap.release()
    return out


def rest_frame(clip: Path, frames: list[np.ndarray]) -> np.ndarray:
    """A frame of this clip in which MediaPipe found no hand (hands lowered), else a blank frame.

    The pause between sentences must look like a signer at rest. Freezing the clip's last frame
    is wrong when the clip ends with the hands still raised: the demo then sees a held sign.
    """
    lm = DATA / "landmarks" / f"{clip.stem}.npz"
    if lm.exists():
        hp = np.load(lm)["hand_present"]
        act = np.flatnonzero(hp.any(axis=1))
        idle = np.flatnonzero(~hp.any(axis=1))
        idle = idle[idle < len(frames)]
        if len(idle):
            if len(act) == 0:
                return frames[int(idle[0])]
            # the idle frame farthest in time from any frame with a hand: a hand that is just
            # leaving the view is often re-detected after video re-encoding (flicker)
            dist = np.abs(idle[:, None] - act[None, :]).min(axis=1)
            return frames[int(idle[int(np.argmax(dist))])]
    return np.full((H, W, 3), 128, np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-signers", type=int, default=4)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(DATA / "subset.csv", encoding="utf-8")) if r["split"] == args.split]
    by_signer: dict[str, dict[str, str]] = {}
    for r in rows:
        by_signer.setdefault(r["Participant ID"], {})[r["Gloss"]] = Path(r["Video file"]).name
    need = {g for s in SCRIPT for g in s}
    signers = sorted(s for s, m in by_signer.items() if need <= set(m))
    print(f"{args.split} signers with all {len(need)} signs: {len(signers)} of {len(by_signer)}")
    out = DATA / f"demo_{args.split}"
    out.mkdir(exist_ok=True)
    made = []
    for s in signers[: args.max_signers]:
        frames: list[np.ndarray] = []
        for sent in SCRIPT:
            for g in sent:
                clip = DATA / "videos" / by_signer[s][g]
                cf = read_frames(clip)
                frames += cf
            frames += [rest_frame(clip, cf)] * int(3 * FPS)  # 3 s at rest between sentences
        path = out / f"conversation_{s}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
        made.append({"signer": s, "video": path.name, "sentences": SCRIPT, "seconds": round(len(frames) / FPS, 1)})
        print(f"wrote {path.name} ({len(frames) / FPS:.0f} s)")
    (out / "ground_truth.json").write_text(json.dumps(made, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
