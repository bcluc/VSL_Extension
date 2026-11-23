"""Live VSL fingerspelling: hold a letter still to type it.

    python -X utf8 scripts/live_letter_demo.py
    python -X utf8 scripts/live_letter_demo.py --threshold 0.7 --hold 1.0

Needs a model trained with scripts/train_letter_classifier.py. "Mũ" after A/E/O gives Â/Ê/Ô,
"Râu" after O/U gives Ơ/Ư. Lower the hand (or change letter) to type the same letter twice.

Keys: SPACE space, BACKSPACE delete, C clear, Q / ESC quit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.letters.labels import display  # noqa: E402
from signbridge.letters.recognizer import LetterRecognizer, LetterTyper  # noqa: E402
from signbridge.live_io import LatestFrame, draw_hand, open_camera, text_panel  # noqa: E402

MODELS = REPO.parent / "models"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=MODELS / "letters" / "vsl_letters.pt")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--hold", type=float, default=0.8, help="seconds a letter must stay stable")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    try:
        rec = LetterRecognizer(args.ckpt, MODELS / "mediapipe" / "hand_landmarker.task")
    except FileNotFoundError as e:
        sys.exit(str(e))
    typer = LetterTyper(rec.labels, threshold=args.threshold, hold_s=args.hold)
    cap = open_camera(args.camera)
    if not cap.isOpened():
        sys.exit("Cannot open the camera.")
    grab = LatestFrame(cap)
    grab.start()
    seq, fps, t_fps, n_fps = 0, 0.0, time.time(), 0
    try:
        while True:
            frame, _, seq = grab.get(seq)
            if frame is None:
                break
            view = cv2.flip(frame, 1)
            pred = rec.predict(view)
            now = time.time()
            typer.push(pred, now)
            n_fps += 1
            if now - t_fps >= 0.5:
                fps, t_fps, n_fps = n_fps / (now - t_fps), now, 0

            if pred.hand is not None:
                draw_hand(view, pred.hand.landmarks[:, :2])
            cur, score = typer.current()
            top = "  ".join(f"{display(lb)} {p:.0%}" for lb, p in pred.top(3)) or "no hand"
            lines = ((f"{display(cur) if cur else '—'}   ({score:.0%})", (120, 220, 255), 44),
                     (f"top 3: {top}", (200, 200, 200), 22),
                     (f"Text: {typer.text}", (255, 255, 255), 30),
                     (f"{fps:.1f} fps · {pred.ms:.0f} ms · SPACE space · BACKSPACE delete · C clear · Q quit",
                      (150, 150, 150), 18))
            h, w = view.shape[:2]
            cv2.imshow("VSL letters", np.vstack([view, text_panel(lines, w, 160)]))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                typer.space()
            elif k == 8:
                typer.backspace()
            elif k == ord("c"):
                typer.clear()
    finally:
        grab.stop()
        cap.release()
        rec.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
