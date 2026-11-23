"""Download the MediaPipe Tasks models used by SignBridge into ../models/mediapipe (outside Git).

    python scripts/fetch_mediapipe_models.py

Models are published by Google under the Apache License 2.0.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[2] / "models" / "mediapipe"
BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        out = DEST / name
        if out.exists() and out.stat().st_size > 0:
            print(f"have {out}")
            continue
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(out)
        print(f"downloaded {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
