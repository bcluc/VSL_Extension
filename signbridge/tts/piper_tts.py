"""Offline Vietnamese TTS with Piper (https://github.com/OHF-Voice/piper1-gpl).

* Engine: ``piper-tts`` (GPL-3.0-or-later), ONNX Runtime on CPU, espeak-ng phonemizer
  bundled (Vietnamese numbers are read as words by espeak-ng).
* Default voice: ``vi_VN-vais1000-medium`` from rhasspy/piper-voices, 22.05 kHz, one
  speaker, trained on the VAIS-1000 corpus (CC BY 4.0). Fetch it with
  ``python scripts/fetch_tts_voice.py``; it is stored outside the repository.

Nothing is sent over the network at synthesis time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .base import AudioClip, TextToSpeech, TTSError, TTSUnavailableError

DEFAULT_VOICE = "vi_VN-vais1000-medium"
VOICE_ENV = "SIGNBRIDGE_TTS_VOICE"
_REPO = Path(__file__).resolve().parents[2]
DEFAULT_VOICE_DIR = _REPO.parent / "models" / "tts" / "piper"


def default_voice_path() -> Path:
    """``$SIGNBRIDGE_TTS_VOICE`` if set, else ``../models/tts/piper/<DEFAULT_VOICE>.onnx``."""
    env = os.environ.get(VOICE_ENV)
    return Path(env) if env else DEFAULT_VOICE_DIR / f"{DEFAULT_VOICE}.onnx"


class PiperTTS(TextToSpeech):
    def __init__(self, model_path: str | Path | None = None, *, speaker_id: int | None = None,
                 length_scale: float | None = None, sentence_silence_s: float = 0.15,
                 require_vietnamese: bool = True):
        """
        :param model_path: ``.onnx`` voice; its ``.onnx.json`` config must sit next to it.
        :param length_scale: > 1 speaks slower, < 1 faster (voice default if None).
        :param sentence_silence_s: silence inserted between sentences.
        :param require_vietnamese: refuse voices whose config is not Vietnamese.
        """
        self.model_path = Path(model_path) if model_path else default_voice_path()
        config_path = Path(f"{self.model_path}.json")
        if not self.model_path.is_file() or not config_path.is_file():
            raise TTSUnavailableError(
                f"Piper voice not found at {self.model_path} (+ .json). "
                "Run: python scripts/fetch_tts_voice.py")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        lang = (cfg.get("language") or {}).get("code") or (cfg.get("espeak") or {}).get("voice", "")
        if require_vietnamese and not str(lang).lower().startswith("vi"):
            raise TTSUnavailableError(f"voice {self.model_path.name} is {lang!r}, not Vietnamese")
        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as e:
            raise TTSUnavailableError("Piper is not installed: pip install piper-tts") from e

        self._voice = PiperVoice.load(self.model_path)
        self._config = SynthesisConfig(speaker_id=speaker_id, length_scale=length_scale)
        self.sample_rate = int(self._voice.config.sample_rate)
        self.sentence_silence_s = sentence_silence_s
        self.language = str(lang).replace("_", "-")
        self.name = f"piper:{self.model_path.stem}"

    def _synthesize(self, text: str) -> AudioClip:
        try:
            chunks = [c.audio_float_array.astype(np.float32).reshape(-1)
                      for c in self._voice.synthesize(text, self._config)]
        except Exception as e:  # onnxruntime / phonemizer errors
            raise TTSError(f"Piper failed on {text!r}: {e}") from e
        if not chunks:
            return AudioClip(np.zeros(0, np.float32), self.sample_rate)
        gap = np.zeros(int(self.sentence_silence_s * self.sample_rate), np.float32)
        parts: list[np.ndarray] = []
        for i, c in enumerate(chunks):
            if i:
                parts.append(gap)
            parts.append(c)
        return AudioClip(np.concatenate(parts), self.sample_rate)
