"""Engine-independent text-to-speech contract.

TTS is the last stage of SignBridge (translation -> Vietnamese text -> speech) but it does
not depend on CSLR or SLT: it only takes a Vietnamese string. Every engine implements
`TextToSpeech.synthesize(text) -> AudioClip`; text validation and WAV output are shared.
"""
from __future__ import annotations

import io
import re
import sys
import unicodedata
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_TEXT_CHARS = 2000
_WS_RX = re.compile(r"\s+")


class TTSError(RuntimeError):
    """Synthesis failed inside the engine."""


class TTSUnavailableError(TTSError):
    """The engine or its voice model is not installed."""


class InvalidTextError(ValueError):
    """The input cannot be spoken (wrong type, empty, nothing pronounceable, too long)."""


def prepare_text(text: object, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Validate and normalize input text before it reaches an engine.

    * must be a ``str`` (``None``/bytes/numbers are rejected, not coerced)
    * Unicode NFC, so decomposed diacritics (e.g. from macOS or some keyboards) match the
      precomposed Vietnamese letters the phonemizer expects
    * control characters removed, whitespace collapsed
    * must contain at least one letter or digit: punctuation-only input is rejected,
      because engines otherwise read the punctuation names aloud
    """
    if not isinstance(text, str):
        raise InvalidTextError(f"text must be str, got {type(text).__name__}")
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
    text = _WS_RX.sub(" ", text).strip()
    if not text:
        raise InvalidTextError("text is empty")
    if not any(ch.isalnum() for ch in text):
        raise InvalidTextError("text contains no letters or digits")
    if len(text) > max_chars:
        raise InvalidTextError(f"text has {len(text)} characters (limit {max_chars}); split it first")
    return text


@dataclass(frozen=True)
class AudioClip:
    """Mono audio, float32 samples in [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        s = np.asarray(self.samples)
        if s.ndim != 1:
            raise ValueError(f"samples must be 1-D (mono), got shape {s.shape}")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "samples", s.astype(np.float32, copy=False))

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.samples ** 2))) if len(self.samples) else 0.0

    def to_int16(self) -> np.ndarray:
        return (np.clip(self.samples, -1.0, 1.0) * 32767).astype("<i2")

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.to_int16().tobytes())
        return buf.getvalue()

    def save_wav(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_wav_bytes())
        return path

    def play(self) -> None:
        """Play through the default output device and wait until it finishes
        (winsound on Windows, sounddevice elsewhere)."""
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(self.to_wav_bytes(), winsound.SND_MEMORY)
            return
        try:
            import sounddevice as sd
        except ImportError as e:  # pragma: no cover - platform dependent
            raise TTSUnavailableError("install 'sounddevice' for audio playback") from e
        sd.play(self.samples, self.sample_rate)
        sd.wait()


class TextToSpeech(ABC):
    """Contract for a Vietnamese TTS engine."""

    #: BCP-47 language of the loaded voice, e.g. "vi-VN"
    language: str = "vi-VN"
    #: human-readable engine/voice name
    name: str = "tts"

    def synthesize(self, text: str) -> AudioClip:
        clean = prepare_text(text)
        clip = self._synthesize(clean)
        if not len(clip.samples):
            raise TTSError(f"{self.name} produced no audio for {clean!r}")
        if not np.isfinite(clip.samples).all():
            raise TTSError(f"{self.name} produced non-finite samples")
        return clip

    def synthesize_to_file(self, text: str, path: str | Path) -> Path:
        return self.synthesize(text).save_wav(path)

    @abstractmethod
    def _synthesize(self, text: str) -> AudioClip:
        """Engine-specific synthesis of already-validated text."""
