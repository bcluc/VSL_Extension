"""Download a Piper Vietnamese voice (ONNX model + JSON config + model card) and verify it.

Voices come from the official catalogue https://huggingface.co/rhasspy/piper-voices.
Files are checked against the MD5 digests published in that catalogue's voices.json.
They are stored OUTSIDE the repository (default: ../models/tts/piper) and must not be
committed.

    python scripts/fetch_tts_voice.py                          # vi_VN-vais1000-medium
    python scripts/fetch_tts_voice.py --voice vi_VN-vivos-x_low
    python scripts/fetch_tts_voice.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
CATALOGUE = BASE + "voices.json"
DEFAULT_VOICE = "vi_VN-vais1000-medium"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "models" / "tts" / "piper"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "signbridge-research"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--list", action="store_true", help="list the Vietnamese voices in the catalogue")
    args = ap.parse_args()

    catalogue = json.loads(_get(CATALOGUE))
    if args.list:
        for key, entry in sorted(catalogue.items()):
            if entry["language"]["code"].startswith("vi"):
                size = sum(f["size_bytes"] for f in entry["files"].values())
                print(f"{key:28s} quality={entry['quality']:7s} speakers={entry['num_speakers']} {size / 1e6:.1f} MB")
        return 0
    if args.voice not in catalogue:
        print(f"unknown voice {args.voice!r}; use --list", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    for rel, meta in catalogue[args.voice]["files"].items():
        name = Path(rel).name
        dest = args.out / (f"{args.voice}.{name}" if name == "MODEL_CARD" else name)
        if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == meta["md5_digest"]:
            print(f"ok (cached) {dest.name}")
            continue
        data = _get(BASE + rel)
        digest = hashlib.md5(data).hexdigest()
        if len(data) != meta["size_bytes"] or digest != meta["md5_digest"]:
            print(f"checksum mismatch for {rel}: got {digest}, expected {meta['md5_digest']}", file=sys.stderr)
            return 1
        dest.write_bytes(data)
        print(f"ok {dest.name} ({len(data) / 1e6:.1f} MB, md5 {digest})")
    print(f"voice stored in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
