"""Vietnamese TTS. Contract tests use a fake engine (no dependencies); tests marked `tts`
run the real Piper voice and are skipped when it is not installed/downloaded."""
import io
import wave

import numpy as np
import pytest

from signbridge.tts import (AudioClip, InvalidTextError, TextToSpeech, TTSError,
                            TTSUnavailableError, load_tts, prepare_text)

SENTENCES = [
    "Xin chào, tôi tên là Lan.",
    "Hôm nay trời đẹp quá.",
    "Bạn có khỏe không?",
    "Tôi đang học ngôn ngữ ký hiệu Việt Nam.",
    "Cảm ơn bạn rất nhiều. Hẹn gặp lại!",
    "Lớp học bắt đầu lúc 7 giờ 30 phút.",
]


# ---------------------------------------------------------------- input validation
@pytest.mark.parametrize("bad", [None, 123, b"xin chao", ["xin chào"]])
def test_non_string_rejected(bad):
    with pytest.raises(InvalidTextError, match="must be str"):
        prepare_text(bad)


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "!!!", "... ?", "\x00\x07"])
def test_empty_or_unpronounceable_rejected(bad):
    with pytest.raises(InvalidTextError):
        prepare_text(bad)


def test_too_long_rejected():
    with pytest.raises(InvalidTextError, match="limit"):
        prepare_text("a" * 50, max_chars=10)


def test_nfc_whitespace_and_control_normalization():
    decomposed = "Việt   Nam\n\tđẹp"        # 'Việt' typed with combining marks
    assert prepare_text(decomposed) == "Việt Nam đẹp"
    assert prepare_text("Xin​chào\x07") == "Xin chào"  # format/control chars -> space


# ---------------------------------------------------------------- audio container
def test_audio_clip_wav_roundtrip(tmp_path):
    sr = 16000
    clip = AudioClip(np.sin(np.linspace(0, 440 * 2 * np.pi, sr)).astype(np.float32) * 0.5, sr)
    assert clip.duration == pytest.approx(1.0) and clip.rms > 0.3
    path = clip.save_wav(tmp_path / "a" / "x.wav")
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, sr, sr)
    with wave.open(io.BytesIO(clip.to_wav_bytes())) as w:
        assert w.getnframes() == sr


def test_audio_clip_rejects_bad_shapes():
    with pytest.raises(ValueError):
        AudioClip(np.zeros((2, 10)), 16000)
    with pytest.raises(ValueError):
        AudioClip(np.zeros(10), 0)


def test_audio_clip_clips_out_of_range_samples():
    assert AudioClip(np.array([2.0, -2.0]), 8000).to_int16().tolist() == [32767, -32767]


# ---------------------------------------------------------------- engine contract (fake engine)
class _FakeTTS(TextToSpeech):
    name = "fake"

    def __init__(self, output=None):
        self.seen, self.output = [], output

    def _synthesize(self, text):
        self.seen.append(text)
        if self.output is not None:
            return AudioClip(self.output, 8000)
        return AudioClip(np.full(80 * len(text), 0.1, np.float32), 8000)


def test_engine_receives_only_validated_text():
    tts = _FakeTTS()
    tts.synthesize("  Xin   chào ")
    assert tts.seen == ["Xin chào"]
    with pytest.raises(InvalidTextError):
        tts.synthesize("   ")
    assert tts.seen == ["Xin chào"]          # invalid input never reaches the engine


def test_engine_empty_or_nan_output_is_an_error():
    with pytest.raises(TTSError, match="no audio"):
        _FakeTTS(np.zeros(0, np.float32)).synthesize("xin chào")
    with pytest.raises(TTSError, match="non-finite"):
        _FakeTTS(np.array([np.nan], np.float32)).synthesize("xin chào")


def test_synthesize_to_file(tmp_path):
    p = _FakeTTS().synthesize_to_file("Xin chào", tmp_path / "o.wav")
    assert p.exists() and p.stat().st_size > 44


def test_unknown_backend_and_missing_voice(tmp_path):
    with pytest.raises(ValueError, match="unknown TTS backend"):
        load_tts("nope")
    with pytest.raises(TTSUnavailableError, match="fetch_tts_voice"):
        load_tts("piper", model_path=tmp_path / "missing.onnx")


def test_non_vietnamese_voice_refused(tmp_path):
    model = tmp_path / "en.onnx"
    model.write_bytes(b"")
    (tmp_path / "en.onnx.json").write_text('{"language": {"code": "en_US"}}', encoding="utf-8")
    with pytest.raises(TTSUnavailableError, match="not Vietnamese"):
        load_tts("piper", model_path=model)


# ---------------------------------------------------------------- real Piper voice
@pytest.fixture(scope="module")
def piper_tts():
    try:
        return load_tts("piper")
    except TTSUnavailableError as e:
        pytest.skip(str(e))


@pytest.mark.tts
@pytest.mark.parametrize("sentence", SENTENCES)
def test_piper_speaks_vietnamese_sentences(piper_tts, sentence, tmp_path):
    clip = piper_tts.synthesize(sentence)
    assert clip.sample_rate == 22050 and piper_tts.language == "vi-VN"
    assert 0.5 < clip.duration < 15.0
    assert np.isfinite(clip.samples).all() and clip.rms > 0.01   # audible, not silence
    with wave.open(str(clip.save_wav(tmp_path / "s.wav"))) as w:
        assert w.getnframes() == len(clip.samples)


@pytest.mark.tts
def test_piper_duration_grows_with_text(piper_tts):
    short = piper_tts.synthesize("Xin chào.").duration
    long = piper_tts.synthesize("Xin chào. " + SENTENCES[3] + " " + SENTENCES[4]).duration
    assert long > 2 * short


@pytest.mark.tts
def test_piper_rejects_invalid_input(piper_tts):
    for bad in ["", "   ", "?!", None]:
        with pytest.raises(InvalidTextError):
            piper_tts.synthesize(bad)
