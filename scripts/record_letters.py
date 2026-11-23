"""Record VSL fingerspelling takes from the webcam. Saves hand landmarks only, never images.

    python -X utf8 scripts/record_letters.py --signer s01
    python -X utf8 scripts/record_letters.py --signer s01 --labels A B C --seconds 3

Each take: a 1.5 s countdown, then ``--seconds`` of capture. Only frames with a detected hand
are kept. After a take the next letter is shown; after the last letter it wraps round, so
going round several times gives several takes per letter recorded at different moments
(move your hand, distance and angle a little between rounds). Aim for at least 5 takes per
letter: training holds out whole takes for validation and test.

Keys: SPACE record, N / D next letter, P / A previous letter, X delete the last take of this
letter, Q / ESC quit.
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

from signbridge.letters.labels import DEFAULT_LABELS, display  # noqa: E402
from signbridge.letters.recognizer import HandDetector  # noqa: E402
from signbridge.letters.recordings import count_takes, save_take  # noqa: E402
from signbridge.live_io import LatestFrame, draw_hand, open_camera, text_panel  # noqa: E402

MODELS = REPO.parent / "models"
DEFAULT_ROOT = REPO.parent / "datasets" / "vsl_letters"
COUNTDOWN_S = 1.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True, help="short id for the person recording, e.g. s01 (no names)")
    ap.add_argument("--labels", nargs="+", default=[k for k, _ in DEFAULT_LABELS])
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    det = HandDetector(MODELS / "mediapipe" / "hand_landmarker.task")
    cap = open_camera(args.camera)
    if not cap.isOpened():
        sys.exit("Cannot open the camera.")
    grab = LatestFrame(cap)
    grab.start()

    i, seq = 0, 0
    state, t_state = "idle", 0.0
    buf_lm: list[np.ndarray] = []
    buf_left: list[bool] = []
    n_frames, last_msg = 0, ""
    counts = count_takes(args.root, args.signer)
    print(f"Saving to {args.root / args.signer}")
    try:
        while True:
            frame, _, seq = grab.get(seq)
            if frame is None:
                break
            view = cv2.flip(frame, 1)
            hand = det.detect(view)
            now = time.time()
            label = args.labels[i]

            if state == "countdown" and now - t_state >= COUNTDOWN_S:
                state, t_state, buf_lm, buf_left, n_frames = "recording", now, [], [], 0
            if state == "recording":
                n_frames += 1
                if hand is not None:
                    buf_lm.append(hand.landmarks)
                    buf_left.append(hand.is_left)
                if now - t_state >= args.seconds:
                    if len(buf_lm) >= 0.5 * n_frames and buf_lm:
                        fps = n_frames / max(now - t_state, 1e-3)
                        save_take(args.root, args.signer, label, buf_lm, buf_left,
                                  view.shape[1] / view.shape[0], fps)
                        counts[label] = counts.get(label, 0) + 1
                        last_msg = f"saved {display(label)}: {len(buf_lm)} frames"
                        i = (i + 1) % len(args.labels)
                    else:
                        last_msg = f"discarded: hand seen in only {len(buf_lm)}/{n_frames} frames"
                    state = "idle"

            if hand is not None:
                draw_hand(view, hand.landmarks[:, :2])
            h, w = view.shape[:2]
            if state == "countdown":
                msg, col = f"get ready... {COUNTDOWN_S - (now - t_state):.1f}", (255, 200, 60)
            elif state == "recording":
                msg, col = f"RECORDING {args.seconds - (now - t_state):.1f} s", (255, 80, 80)
            else:
                msg, col = "SPACE record · N/P next/prev · X delete last · Q quit", (200, 200, 200)
            lines = ((f"Letter: {display(label)}   ({counts.get(label, 0)} takes)", (120, 220, 255), 40),
                     (msg, col, 24),
                     (last_msg, (160, 160, 160), 20),
                     (f"signer {args.signer} · {i + 1}/{len(args.labels)} · min takes "
                      f"{min(counts.get(lb, 0) for lb in args.labels)}", (160, 160, 160), 20))
            view = np.vstack([view, text_panel(lines, w, 150)])
            cv2.imshow("Record VSL letters (landmarks only)", view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if state != "idle":
                continue
            if k == ord(" "):
                state, t_state = "countdown", now
            elif k in (ord("n"), ord("d")):
                i = (i + 1) % len(args.labels)
            elif k in (ord("p"), ord("a")):
                i = (i - 1) % len(args.labels)
            elif k == ord("x"):
                takes = sorted((args.root / args.signer / label).glob("take_*.npz"))
                if takes:
                    takes[-1].unlink()
                    counts[label] -= 1
                    last_msg = f"deleted last take of {display(label)}"
    finally:
        grab.stop()
        cap.release()
        det.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
