# SignBridge Research Architecture

Target product: **continuous sign language → Vietnamese text → speech** (CSLT + TTS). This document
describes the dataset-independent research core in [`signbridge/`](../signbridge). It was validated
on How2Sign (ASL) and is designed so a VSL dataset plugs in without changes outside its adapter.

## 1. Applications built on the core

| Component | Role | Code |
|---|---|---|
| **VSL fingerspelling** (static letters) | MediaPipe hand landmarks (21 points) → 83-d hand-shape features → MLP → "hold to type" text, with Mũ/Râu composing Â Ê Ô Ơ Ư. Trained on the user's own recordings (landmarks only, no images). | `signbridge/letters/`, `scripts/record_letters.py`, `train_letter_classifier.py`, `live_letter_demo.py` |
| `teams_app/` (FastAPI + Teams tab) | Serves the letter recognizer to a Teams personal tab | imports `signbridge.letters` |
| **ASL greetings demo** | 29 isolated ASL signs → phrase table → speech | `signbridge/signs/`, `scripts/live_sign_demo.py` |
| MediaPipe Tasks models | hand and pose landmarkers, in `../models/mediapipe/` | `scripts/fetch_mediapipe_models.py` |

The CSLT core below does not depend on the applications. The Teams backend can later call the
CSLT pipeline through the same `FrameSource` → `VideoPreprocessor` → `SignTranslationPipeline` path.

## 2. Data pipeline

```
 native dataset files                 dataset-specific (only here)
 ───────────────────────►  Adapter (How2SignDataset | VSLDataset | ManifestDataset)
                                │  load_manifest() → ManifestRecord[] (common schema, nulls for missing)
                                │  get_sample() / load_annotations() / get_metadata()
                                ▼  load_video() → FrameSource (lazy)
 ─────────────────────────── dataset-agnostic from here ───────────────────────────
 FrameSource (VideoReader: mp4 + [start,end) | ImageFolderReader: jpg sequence)
                                │  sample_indices(): all | uniform | stride | fps(resample)
                                ▼  sequential decode, undecodable frame → None
 VideoPreprocessor (PreprocessConfig, JSON-configurable)
   • missing-frame policy: repeat_last | zero | drop (+ valid_mask, max_missing_ratio)
   • resize (letterbox by default) → RGB → normalize → float32 (T,C,H,W)
   • process(): whole clip, capped by max_frames
   • stream_windows(): fixed windows, memory bounded by one window
                                ▼
 collate(): pad variable T → (B,T,…) + lengths + mask + frame_valid
            gloss ids / text ids only from real annotations; has_gloss / has_text flags
                                ▼
 SignTranslationPipeline (models, §4)
```

### 2.1 Common manifest schema ([`signbridge/data/schema.py`](../signbridge/data/schema.py))

`sample_id, dataset_name, language, video_path, signer_id?, gloss_sequence?, source_text?,
target_text?, start_time?, end_time?, fps?, split, annotation_status`, plus a free-form `meta`
dict for provenance. Rules:

* Missing values are `null`. Adapters never invent values: `fps` is *measured* from the file
  when it exists, otherwise `null`.
* `start_time`/`end_time` are seconds relative to `video_path`. `null` means the whole file.
* `gloss_sequence` is filled only from real gloss annotations of the same sign language.
* `annotation_status` ∈ `gloss_and_translation | gloss_only | translation_only | isolated_label | unannotated`.
  Training code chooses losses from this field and never guesses.
* `language` uses project ids `asl`/`vsl`. ISO 639-3 `vsl` is *Venezuelan* SL, so it is never
  used as an ISO code. Vietnamese sign languages have regional codes (`hab`, `haf`, `hos`).
* Serialized as JSONL (canonical) plus a CSV view for inspection.

### 2.2 Dataset adapters

| Adapter | Implemented layouts | Notes |
|---|---|---|
| `How2SignDataset(root, split, view)` | official How2Sign tree | views `clips`, `original_full`, `realigned_full` (see [dataset_audit §8](dataset_audit.md#8-temporal-boundaries-original-vs-re-aligned)) |
| `VSLDataset(root, layout="letter_frames")` | `<root>/<LABEL_ID>/frame_*.jpg` image folders | isolated letters. The label comes from the folder name via `signbridge.letters.labels` (`DD` = Đ, `MU` = mũ, `RAU` = râu). `fps` is `null` (not recorded). Tested on synthetic fixtures only. |
| `VSLDataset(root, layout="continuous_csv")` | **proposed** `manifest.csv` | columns `sample_id, video_path, split, signer_id, gloss, vietnamese_text, start_time, end_time`. Tested on synthetic fixtures only; no continuous VSL data exists yet. |
| `ManifestDataset(path)` | any common-schema JSONL | lets external tooling or annotation exports feed the pipeline directly |

All adapters subclass `SignDataset` and implement a single method, `_build_records()`. The public
methods `load_manifest`, `get_sample`, `load_video`, `load_annotations` and `get_metadata` are
shared. Errors are typed: `VideoUnavailableError`, `SampleNotFoundError`, `AnnotationMissingError`.

## 3. Temporal processing

* **Readers stream.** `VideoReader.read_frames(indices)` decodes sequentially and `grab()`s skipped
  frames. It never holds the whole video in memory. It has two seek modes: `seek` (container seek)
  and `grab` (frame-exact decode from frame 0). On the real H.264 full video, both returned
  byte-identical frames.
* **Mixed frame rates** (How2Sign has 24, 30 and 60 fps) are normalized with `sampling="fps"`
  (default 12 fps), so one timestep means the same duration for every source. Validated: 11.2–12.0
  steps per second of video across all 34 real clips. Short clips fall below 12 because of rounding.
* **Windows**: `make_windows(T, window, stride)` gives overlapping `[start, end)` windows with a
  padded last window and a mask. This serves sliding-window inference on live camera streams and
  long videos.
* **Long videos**: `stream_windows()` keeps at most one window of processed frames. On the 46 s
  full video it measured **31.8 MB** peak traced memory, versus **2,191 MB** for whole-clip processing.
* **Missing frames** are explicit (`valid_mask`). Above `max_missing_ratio` a `CorruptVideoError`
  is raised instead of silently training on garbage.
* **Video ↔ annotation validation** (`validate_sample`) checks:
  * the file exists and decodes,
  * FPS and frame count are sane,
  * the segment lies inside the video,
  * the annotation matches the declared status,
  * clip duration agrees with the annotated interval.

  `locate_clip_in_video` measures real temporal offsets between released clips and their source
  videos.

## 4. Model interfaces ([`signbridge/models/interfaces.py`](../signbridge/models/interfaces.py))

```
inputs (B,T,C,H,W) rgb | (B,T,K,C) pose, lengths (B,)
  → VisualEncoder        → SequenceFeatures(features B×T×D, lengths, mask)
  → TemporalEncoder      → SequenceFeatures(B×T'×D'), may downsample (lengths updated)
  ├→ GlossDecoder (CTC)  → GlossOutput(log_probs T'×B×V, lengths, blank_id); loss(); decode()
  └→ TranslationModel    → logits B×L×V (teacher forcing); generate() → token ids
```

* **Replaceable encoders.** RGB (`FrameCNNEncoder`, `TorchvisionFrameEncoder`) and pose
  (`PoseMLPEncoder` over MediaPipe `(T, 42, 4)` = 2 hands × 21 keypoints × (x, y, z, present))
  satisfy the same `VisualEncoder` contract. Padding never reaches the encoder: only `mask == True`
  frames are encoded, and a unit test confirms garbage in the padding does not change the output.
* **Gloss recognition interface (CSLR).** `CTCGlossDecoder` follows the standard CTC convention:
  blank = 0, greedy collapse decoding, and `nn.CTCLoss` with input and target lengths.
  `GlossVocabulary.from_sequences` **refuses to build an empty vocabulary**. On How2Sign this
  refusal is the expected behavior, because the dataset has no glosses.
* **Translation interface.** `TranslationModel.forward(memory, target_in)` and `generate(memory)`.
  `SignTranslationPipeline` has two modes:
  * **Gloss-free Sign2Text**: translator reads the temporal features. This is the only option
    trainable on How2Sign.
  * **Sign2Gloss2Text / joint CTC + translation**: gloss decoder and translator together. Needs
    gloss labels, which future VSL data should provide.
* The **reference modules are untrained** and exist only to validate shapes, masks, lengths,
  CTC computability and the decoding loop. Production candidates (pretrained video backbones,
  pose GCNs, mBART/LLM decoders for Vietnamese) implement the same ABCs.
* **Text** ([`signbridge/text/vocab.py`](../signbridge/text/vocab.py)): Unicode-aware tokenization
  that keeps Vietnamese diacritics, `<pad>/<unk>/<bos>/<eos>` specials, and vocabularies built
  only from real annotations. A subword tokenizer (e.g. SentencePiece) can replace it behind
  `encode_text`.

Speech output (TTS) sits after `TranslationModel.generate()`. It is implemented
independently in `signbridge/tts/` (offline Piper, Vietnamese voice) and takes plain
Vietnamese text. See [tts_implementation.md](tts_implementation.md).

## 5. How2Sign → VSL migration strategy

1. **Collect continuous VSL** with sentence-level Vietnamese translations. Annotate **VSL glosses**
   too, if possible: glosses are what makes the CTC/CSLR branch trainable. Annotators should
   produce glosses from the VSL signing itself, never by converting Vietnamese text into gloss
   tokens.
2. **Inspect the real annotation format first.** Then either export it to the proposed
   `continuous_csv` layout, or write a `VSL<Format>Dataset._build_records()` that emits
   `ManifestRecord`s. No other module changes. Record `signer_id` from day one so splits can be
   signer-independent.
3. **Run the same checks**: `scripts/build_manifests.py`, `validate_sample` on every record, and the
   validation experiment pointed at the VSL manifest. Fix data issues (FPS, cut boundaries,
   missing frames) before any training.
4. **Choose supervision by `annotation_status`**: CTC loss only where `has_gloss`, translation loss
   only where `has_text`. Mixed datasets work because `collate` reports these flags per sample.
5. **Do not transfer labels across languages.** If How2Sign is used at all for VSL models, use it
   only as *unsupervised or representation pre-training* for visual/pose encoders. That is
   language-agnostic motion modeling, and CC BY-NC limits it to research. No ASL→VSL gloss or text
   mapping.
6. **Serving**: wrap camera frames in a `FrameSource` (same protocol as `VideoReader`) and use
   `stream_windows()` for low-latency sliding-window inference inside the existing FastAPI/Teams
   backend.

## 6. File map

| Path | Content |
|---|---|
| `signbridge/data/schema.py` | `ManifestRecord`, `AnnotationStatus`, JSONL/CSV I/O |
| `signbridge/data/base.py` | `SignDataset` ABC, `Annotation`, `Sample`, `ManifestDataset`, errors |
| `signbridge/data/how2sign.py`, `vsl.py` | dataset adapters (the only dataset-specific code) |
| `signbridge/data/collate.py`, `validation.py` | batching; video ↔ annotation checks |
| `signbridge/video/` | `io.py` readers and probe, `sampling.py`, `transforms.py`, `pipeline.py` |
| `signbridge/features/hand_keypoints.py` | MediaPipe 2-hand keypoints (pose path) |
| `signbridge/text/vocab.py` | gloss and text vocabularies |
| `signbridge/models/` | `interfaces.py` (contracts), `reference.py` (untrained reference modules) |
| `signbridge/tts/` | text-to-speech (`TextToSpeech`, `AudioClip`, Piper Vietnamese backend, Windows SAPI English backend, CLI) |
| `signbridge/metrics/` | WER (sclite weights) and the official PHOENIX-2014-T gloss normalisation |
| `signbridge/signs/` | isolated-sign demo: MediaPipe pose + hands features, GRU classifier, live state machine, phrase table |
| `signbridge/letters/` | VSL fingerspelling: label set, hand-shape features, MLP + checkpoint format, recognizer and typing logic, landmark recordings |
| `signbridge/live_io.py` | newest-frame camera reader, cached Vietnamese text rendering, hand drawing |
| `configs/preprocess_default.json` | default preprocessing config |
| `scripts/` | `fetch_how2sign_subset.py`, `audit_how2sign.py`, `build_manifests.py`, `validate_pipeline.py` |
| `tests/` | 143 tests: 127 without external data, 7 How2Sign (`-m realdata`), 1 PHOENIX (`-m phoenix`) and 8 real-voice TTS (`-m tts`); the marked tests skip when their data is absent |
