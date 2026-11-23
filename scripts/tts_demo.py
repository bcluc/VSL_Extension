"""Synthesize a fixed list of hand-written Vietnamese sentences and record what was measured.

Writes outputs/tts/<nn>.wav and outputs/tts/results.json (duration, synthesis time,
real-time factor, RMS level, espeak-ng phonemes). Audio QUALITY is not measured here:
listen to the WAV files. Invalid inputs are also exercised to record how they are refused.

    python scripts/tts_demo.py [--play]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.tts import InvalidTextError, load_tts  # noqa: E402

SENTENCES = [
    "Xin chào, tôi tên là Lan.",
    "Hôm nay trời đẹp quá.",
    "Bạn có khỏe không?",
    "Tôi đang học ngôn ngữ ký hiệu Việt Nam.",
    "Cảm ơn bạn rất nhiều. Hẹn gặp lại!",
    "Lớp học bắt đầu lúc 7 giờ 30 phút.",
    "Hôm nay là ngày 23 tháng 11 năm 2026.",
    "Người khiếm thính có quyền được tiếp cận thông tin bằng ngôn ngữ ký hiệu.",
]
INVALID = ["", "   ", "!!!", None, 42]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "tts")
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tts = load_tts("piper")
    load_s = time.perf_counter() - t0
    tts.synthesize("Khởi động.")  # warm-up, excluded from timings

    rows = []
    for i, text in enumerate(SENTENCES, 1):
        t0 = time.perf_counter()
        clip = tts.synthesize(text)
        dt = time.perf_counter() - t0
        wav = clip.save_wav(args.out / f"{i:02d}.wav")
        phon = " | ".join("".join(p) for p in tts._voice.phonemize(text))
        rows.append({"id": i, "text": text, "wav": wav.name, "audio_s": round(clip.duration, 3),
                     "synth_s": round(dt, 3), "rtf": round(dt / clip.duration, 3),
                     "rms": round(clip.rms, 4), "phonemes": phon})
        print(f"{i:02d} {clip.duration:5.2f}s audio  {dt:5.2f}s synth  RTF {dt / clip.duration:.2f}")
        if args.play:
            clip.play()

    refused = []
    for bad in INVALID:
        try:
            tts.synthesize(bad)
            refused.append({"input": repr(bad), "refused": False})
        except InvalidTextError as e:
            refused.append({"input": repr(bad), "refused": True, "reason": str(e)})

    result = {
        "engine": tts.name, "language": tts.language, "sample_rate": tts.sample_rate,
        "model_load_s": round(load_s, 3), "platform": platform.platform(),
        "python": platform.python_version(), "sentences": rows, "invalid_inputs": refused,
        "note": "Timings are CPU wall-clock on this machine. Perceptual quality was not measured.",
    }
    (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"invalid inputs refused: {sum(r['refused'] for r in refused)}/{len(refused)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
