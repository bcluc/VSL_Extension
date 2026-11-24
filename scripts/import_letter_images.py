"""Convert a folder of letter images into landmark takes (the record_letters.py format).

    python -X utf8 scripts/import_letter_images.py SRC --signer ext01 --map Dd=DD Rau=RAU mu_=MU

SRC layout: ``SRC/<FOLDER>/*.jpg|*.png``, one folder per letter. Each folder becomes **one take**
of ``--signer``, because consecutive frames of one recording are not independent samples.
``--map`` renames folders to label ids. Only frames with a detected hand are kept, and no
image is copied. Use ``--mirror`` if the images are NOT already mirrored (selfie view).
Respect the license of the source images; keep the result out of Git (it goes to ../datasets).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.letters.recognizer import HandDetector  # noqa: E402
from signbridge.letters.recordings import save_take  # noqa: E402

MODELS = REPO.parent / "models"
DEFAULT_ROOT = REPO.parent / "datasets" / "vsl_letters"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("--signer", required=True)
    ap.add_argument("--map", nargs="*", default=[], help="FOLDER=LABEL_ID renames")
    ap.add_argument("--mirror", action="store_true", help="flip images horizontally first")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    rename = dict(m.split("=", 1) for m in args.map)
    if (args.root / args.signer).exists():
        sys.exit(f"{args.root / args.signer} already exists; choose another --signer or delete it")
    det = HandDetector(MODELS / "mediapipe" / "hand_landmarker.task")
    try:
        for folder in sorted(p for p in args.src.iterdir() if p.is_dir()):
            files = sorted([*folder.glob("*.jpg"), *folder.glob("*.png")])
            lms, left, aspect = [], [], None
            for f in files:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                if args.mirror:
                    img = cv2.flip(img, 1)
                aspect = img.shape[1] / img.shape[0]
                hand = det.detect(img)
                if hand is not None:
                    lms.append(hand.landmarks)
                    left.append(hand.is_left)
            label = rename.get(folder.name, folder.name)
            if not lms:
                print(f"{folder.name}: no hand found in {len(files)} images, skipped")
                continue
            save_take(args.root, args.signer, label, lms, left, aspect, fps=0.0)
            print(f"{folder.name} -> {label}: {len(lms)}/{len(files)} frames with a hand", flush=True)
    finally:
        det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
