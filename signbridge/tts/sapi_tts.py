"""English speech through the Windows built-in SAPI voices (offline, no download).

Used by the ASL greetings demo for English output. Needs Windows and pywin32.
The installed voices on the development machine were "Microsoft David Desktop" and
"Microsoft Zira Desktop" (en-US).
"""
from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path

import numpy as np

from .base import AudioClip, TextToSpeech, TTSError, TTSUnavailableError

_SAFT22kHz16BitMono = 22  # SpeechAudioFormatType
_SSFMCreateForWrite = 3


class SapiTTS(TextToSpeech):
    def __init__(self, voice: str = "Zira", rate: int = 0):
        try:
            import win32com.client  # type: ignore
        except ImportError as e:  # pragma: no cover - platform specific
            raise TTSUnavailableError("SAPI needs Windows and pywin32") from e
        self._client = win32com.client
        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        voices = [self._voice.GetVoices().Item(i) for i in range(self._voice.GetVoices().Count)]
        match = [v for v in voices if voice.lower() in v.GetDescription().lower()]
        if not match:
            raise TTSUnavailableError(f"no SAPI voice matching {voice!r}; installed: "
                                      f"{[v.GetDescription() for v in voices]}")
        self._voice.Voice = match[0]
        self._voice.Rate = rate
        self.name = f"sapi:{match[0].GetDescription()}"
        self.language = "en-US"
        self.sample_rate = 22050

    def _synthesize(self, text: str) -> AudioClip:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            stream = self._client.Dispatch("SAPI.SpFileStream")
            fmt = self._client.Dispatch("SAPI.SpAudioFormat")
            fmt.Type = _SAFT22kHz16BitMono
            stream.Format = fmt
            stream.Open(tmp, _SSFMCreateForWrite)
            prev = self._voice.AudioOutputStream
            self._voice.AudioOutputStream = stream
            try:
                self._voice.Speak(text)
            finally:
                stream.Close()
                self._voice.AudioOutputStream = prev
            with wave.open(tmp, "rb") as w:
                raw = w.readframes(w.getnframes())
                sr = w.getframerate()
            samples = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
            return AudioClip(samples, sr)
        except TTSError:
            raise
        except Exception as e:  # noqa: BLE001
            raise TTSError(f"SAPI synthesis failed: {e}") from e
        finally:
            Path(tmp).unlink(missing_ok=True)
