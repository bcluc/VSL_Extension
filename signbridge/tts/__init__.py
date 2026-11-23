"""Vietnamese text-to-speech, independent of sign recognition and translation.

    from signbridge.tts import load_tts
    tts = load_tts()                       # offline Piper, vi_VN-vais1000-medium
    tts.synthesize("Xin chào").save_wav("xin_chao.wav")
"""
from .base import (MAX_TEXT_CHARS, AudioClip, InvalidTextError, TextToSpeech, TTSError,
                   TTSUnavailableError, prepare_text)


def load_tts(backend: str = "piper", **kwargs) -> TextToSpeech:
    """Create a TTS engine by name: ``"piper"`` (Vietnamese) or ``"sapi"`` (Windows English voices)."""
    if backend == "piper":
        from .piper_tts import PiperTTS
        return PiperTTS(**kwargs)
    if backend == "sapi":
        from .sapi_tts import SapiTTS
        return SapiTTS(**kwargs)
    raise ValueError(f"unknown TTS backend {backend!r} (available: 'piper', 'sapi')")


__all__ = ["AudioClip", "InvalidTextError", "MAX_TEXT_CHARS", "TextToSpeech", "TTSError",
           "TTSUnavailableError", "load_tts", "prepare_text"]
