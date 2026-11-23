# Vietnamese Text-to-Speech (TTS) Implementation

**Date:** 2026-11-23. **Status:** implemented and tested with hand-written Vietnamese
sentences. TTS is independent of CSLR and SLT: it only takes a Vietnamese string.

## 1. Engine selection

Checked on this machine before choosing:

| Option | Offline | Vietnamese voice | License | Result |
|---|---|---|---|---|
| Windows SAPI / OneCore (`System.Speech`) | yes | **none installed** (only en-US, zh-CN voices) | OS component | Not usable without installing a Windows language pack, which changes system settings. Not done. |
| **Piper** (`piper-tts` 1.8.0) + `vi_VN-vais1000-medium` | **yes** | yes, 22.05 kHz, 1 speaker | engine **GPL-3.0-or-later**; voice dataset VAIS-1000 **CC BY 4.0** | **Selected** |
| Piper `vi_VN-vivos-x_low` | yes | yes, 16 kHz, 65 speakers | dataset VIVOS **CC BY-NC-SA 4.0** | Not default (lower quality, non-commercial) |
| Piper `vi_VN-25hours_single-low` | yes | yes, 16 kHz | dataset license **unknown** (model card) | Rejected: unclear license |
| Cloud services (gTTS, edge-tts, Azure, Google) | no | yes | service terms | Not used: they send the text to an external service and need network access |

Voice facts above come from the model cards and `voices.json` in the official catalogue
<https://huggingface.co/rhasspy/piper-voices> (read 2026-11-23). The default voice was
fine-tuned from the English `hifi_tts_92` Piper voice (model card).

**License consequences.**
* Piper is GPL-3.0-or-later. Using it for research is fine. Distributing a product that links
  it would put that product under GPL obligations. Either keep TTS behind a process
  boundary (e.g. a separate service or the `piper` CLI) or swap the backend.
* VAIS-1000 is CC BY 4.0: attribution is required, and commercial use is allowed.
* Each voice has its own license. Check its `MODEL_CARD` before switching voices.

## 2. Installation

```powershell
pip install "piper-tts>=1.5"
```

```powershell
python scripts/fetch_tts_voice.py
```

The script downloads the `.onnx` model (63.2 MB), its `.onnx.json` config and the model card
into `../models/tts/piper/`, **outside the repository**. It verifies size and MD5 against
the catalogue. Use `--list` to see the Vietnamese voices and `--voice <name>` to pick another.
`SIGNBRIDGE_TTS_VOICE=<path.onnx>` overrides the default voice path.

Dependencies installed by pip: `onnxruntime` 1.32.0 (MIT), `pathvalidate`, `protobuf`.
espeak-ng data for phonemization is bundled inside the Piper wheel. Nothing goes over the
network during synthesis.

## 3. Python interface ([`signbridge/tts/`](../signbridge/tts))

```python
from signbridge.tts import load_tts, InvalidTextError, TTSUnavailableError

tts = load_tts()                                  # PiperTTS with the default voice
clip = tts.synthesize("Xin chào, tôi tên là Lan.")
clip.save_wav("xin_chao.wav")                     # 16-bit PCM mono WAV
clip.play()                                       # default speaker (winsound on Windows)
print(clip.duration, clip.sample_rate)            # seconds, 22050
```

| Item | Contract |
|---|---|
| `TextToSpeech.synthesize(text) -> AudioClip` | Validates the text, calls the engine, then rejects empty or non-finite audio with `TTSError` |
| `prepare_text(text)` | Rejects non-`str` input. Applies Unicode **NFC**, so decomposed diacritics become precomposed letters. Replaces control and format characters with spaces and collapses whitespace. Rejects empty text, text with no letter or digit (e.g. `"!!!"`, which espeak-ng would otherwise read aloud as "chấm than"), and text over 2,000 characters. Errors are `InvalidTextError` (a `ValueError`). |
| `AudioClip(samples, sample_rate)` | Mono float32 in [-1, 1]. Provides `duration`, `rms`, `to_int16()`, `to_wav_bytes()`, `save_wav(path)` and `play()`. |
| `PiperTTS(model_path, speaker_id, length_scale, sentence_silence_s=0.15)` | Refuses a voice whose config is not Vietnamese. Raises `TTSUnavailableError` if Piper or the voice file is missing. Inserts silence between sentences. |
| `load_tts(backend="piper")` | Factory. Other engines (e.g. a commercially licensed one) implement `TextToSpeech._synthesize` |
| CLI | `python -m signbridge.tts "Xin chào các bạn" -o out.wav [--play] [--rate 1.1]`. Exit code 2 for invalid input, 1 for an engine error. |

## 4. Test results (measured on this machine)

**Automated tests:** `tests/test_tts.py` has 28 tests, and **28 pass**.

* 20 need no engine: input validation, NFC, WAV round trip, the fake-engine contract,
  missing voice, and refusal of non-Vietnamese voices.
* 8 are marked `tts` and use the real voice: six Vietnamese sentences produce audible,
  finite 22.05 kHz audio that is written to valid WAV files; duration grows with text
  length; invalid input is refused. The `tts` tests skip when the voice is absent.

**Manual sentences** (`python scripts/tts_demo.py`): this writes `outputs/tts/01.wav` to
`08.wav` and `results.json`. CPU only (Windows 11, Python 3.14.2, onnxruntime 1.32.0).
Model load took 0.68 s.

| # | Sentence | Audio (s) | Synth (s) | RTF |
|---|---|---|---|---|
| 1 | Xin chào, tôi tên là Lan. | 1.49 | 0.18 | 0.12 |
| 2 | Hôm nay trời đẹp quá. | 1.09 | 0.21 | 0.20 |
| 3 | Bạn có khỏe không? | 0.84 | 0.79 | 0.95 |
| 4 | Tôi đang học ngôn ngữ ký hiệu Việt Nam. | 1.71 | 1.70 | 1.00 |
| 5 | Cảm ơn bạn rất nhiều. Hẹn gặp lại! | 1.88 | 2.05 | 1.09 |
| 6 | Lớp học bắt đầu lúc 7 giờ 30 phút. | 1.86 | 0.32 | 0.17 |
| 7 | Hôm nay là ngày 23 tháng 11 năm 2026. | 3.81 | 0.33 | 0.09 |
| 8 | Người khiếm thính có quyền được tiếp cận thông tin bằng ngôn ngữ ký hiệu. | 3.37 | 0.38 | 0.11 |

* **Speed.** RTF is synthesis time divided by audio duration. It ranged from 0.09 to 1.09 in
  this run. In a second run (`outputs/tts_run2/`) it ranged from 0.12 to 0.52, and the
  sentences that were slow in run 1 (3–5) took 0.24, 0.52 and 0.40. So the per-sentence
  timing varies between runs on this machine. The cause was not isolated (background load is
  likely). In both runs every sentence except run 1's sentence 5 synthesized at least as fast
  as real time.
* **Numbers.** espeak-ng phonemes show numbers read as Vietnamese words. "2026" became
  *hai ngàn không trăm hai mươi sáu*; "30" became *ba mươi*.
* **Invalid inputs.** All 5 were refused before reaching the engine: `""`, `"   "`, `"!!!"`,
  `None`, `42`.

**Not measured:**

* Perceptual quality (MOS), naturalness and intelligibility. Nobody has listened to these
  files and rated them, and no ASR round trip was run. **Listen to `outputs/tts/*.wav`
  before relying on the voice.**
* Determinism. Piper/VITS samples noise, so the same sentence can come out slightly
  different (1.46 s vs 1.52 s in two runs).
* Text normalization beyond espeak-ng's own: abbreviations, dates written as `23/11`,
  units, foreign words. For example, "OK" is phonemized as English.

## 5. Next steps for TTS

1. Listen to the 8 WAV files and rate them (5-point MOS, ≥ 3 listeners if possible).
2. Optional objective check: an ASR round trip with a Vietnamese ASR model (e.g. a Whisper
   variant), computing CER against the input text.
3. Add Vietnamese text normalization for dates, abbreviations and units before
   `synthesize`, if the translation output contains them.
4. Product use: decide on GPL (isolate Piper in its own process) versus another engine
   behind the same `TextToSpeech` interface.
