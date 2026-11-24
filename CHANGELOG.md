# Changelog

## v0.1.0 (2026-11-23): first research release

A research prototype, not a product. Every number below was measured offline. **No model has
been measured with a live webcam user.**

### VSL fingerspelling (`signbridge/letters/`)

* **Pipeline:** MediaPipe hand landmarks (21 points) → 83 hand-shape features → small MLP →
  "hold to type" text.
  * The features do not depend on where the hand is, how big it appears, or which hand signs.
  * Mũ / Râu compose Â Ê Ô / Ơ Ư.
* **`scripts/record_letters.py`** records takes from the webcam. It stores hand landmarks only,
  never images.
* **`scripts/train_letter_classifier.py`** holds out whole takes, or a whole signer with
  `--test-signer`, and writes a report with per-letter accuracy and confusions.
* **Live use:**
  * `scripts/live_letter_demo.py` runs on the webcam;
  * the Teams tab (`teams_app/`) serves the same recognizer;
  * both refuse to recognise without a trained model.
* **`scripts/import_letter_images.py`** converts image folders to landmark takes;
  `--train-only-signer` keeps such a signer in training only.
* **First offline results** with 2 signers (25 static letters; details in
  [docs/ai_validation_status.md](docs/ai_validation_status.md) §3):
  * about 50 % frame top-1 on an unseen signer;
  * 68.9 % on a later take of a training signer.
  * Live accuracy has not been measured.

### ASL greetings demo (`signbridge/signs/`, `scripts/live_sign_demo.py`)

* **Recognition and output:** 29 isolated ASL greeting signs from webcam pose and hand
  landmarks, composed into an English or Vietnamese sentence with a fixed phrase table (not a
  translation model), and spoken aloud (Windows SAPI for English, Piper for Vietnamese).
* **Training data:** ASL Citizen (non-commercial research license). The model is not
  distributed.
* **Offline results** ([docs/asl_greetings_demo.md](docs/asl_greetings_demo.md)):
  * 11 held-out signers: test top-1 83.1 % (mean of 3 seeds), word-level 88.4 %;
  * scripted test conversations with the current defaults: WER 25.0 %, 53/90 sentences exact.
* **Live loop:**
  * the camera thread keeps only the newest frame;
  * text rendering is cached;
  * pose and hands are detected in parallel;
  * about 19 fps on CPU.
* **Rest line** drawn on screen, with a warning when a sign is cut at 4 s.
* **Diagnostics:** `--log` and `--save-landmarks` record a session;
  `scripts/tune_live_config.py` replays it.

### Research core

* **Data:** dataset-independent manifests, video I/O, preprocessing and windows; How2Sign
  adapter; VSL adapters (`letter_frames`, proposed `continuous_csv`).
* **Models:** untrained CSLR / SLT reference models, shape-checked only.
* **WER metric** (`signbridge/metrics/`): reproduces the official PHOENIX-2014-T sclite output
  exactly.
* **Vietnamese TTS** (`signbridge/tts/`): offline Piper voice. Perceptual quality has not been
  rated.
* **Docs:** dataset audits, architecture, validation report, data requirements for CSLR and SLT,
  AI validation status, roadmap.

### Not done

* No continuous sign recognition (CSLR), no sign-to-Vietnamese translation (SLT), no VSL
  gloss data. See [docs/next_development_roadmap.md](docs/next_development_roadmap.md).

### Tests

* 146 tests, all passing on 2026-11-24 (`python -m pytest`).
* The tests that need How2Sign, PHOENIX or the Piper voice are skipped when those are absent.
