"""Command line: python -m signbridge.tts "Xin chào các bạn" -o out.wav [--play]"""
from __future__ import annotations

import argparse
import sys
import time

from . import InvalidTextError, TTSError, load_tts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m signbridge.tts", description="Vietnamese text-to-speech")
    ap.add_argument("text", help="Vietnamese text")
    ap.add_argument("-o", "--out", default="tts_output.wav", help="WAV file to write")
    ap.add_argument("--voice", default=None, help="path to a Piper .onnx voice")
    ap.add_argument("--rate", type=float, default=None, help="length scale: >1 slower, <1 faster")
    ap.add_argument("--play", action="store_true", help="play the audio after writing it")
    args = ap.parse_args(argv)
    try:
        tts = load_tts("piper", model_path=args.voice, length_scale=args.rate)
        t0 = time.perf_counter()
        clip = tts.synthesize(args.text)
        dt = time.perf_counter() - t0
    except InvalidTextError as e:
        print(f"invalid input: {e}", file=sys.stderr)
        return 2
    except TTSError as e:
        print(f"TTS error: {e}", file=sys.stderr)
        return 1
    path = clip.save_wav(args.out)
    print(f"{path} | {clip.duration:.2f} s audio | synthesized in {dt:.2f} s | {tts.name}")
    if args.play:
        clip.play()
    return 0


if __name__ == "__main__":
    sys.exit(main())
