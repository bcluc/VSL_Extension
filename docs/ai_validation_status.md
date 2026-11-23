# AI Validation Status

**Date:** 2026-11-23. This file separates four things:

* code that exists;
* code exercised only on synthetic data;
* code exercised on real data;
* models whose **outputs** were checked against real labels.

Tensor-shape, mask and loss-is-finite checks are counted as **interface tests**, never as
model validation.

Sources: the code in `signbridge/`, `tests/` (143 tests, all passing on 2026-11-23),
[validation_report.md](validation_report.md) and [tts_implementation.md](tts_implementation.md).

## 1. Summary

| Stage | Implemented | Synthetic tests | Real-data tests | Meaningful model output validated |
|---|---|---|---|---|
| Video ingestion and preprocessing | yes | yes | yes (34 How2Sign clips, 1 full video) | n/a (no model) |
| Hand keypoints (MediaPipe) | yes | — | yes: runs on 4 real clips, detection ratio measured | **no**: keypoint accuracy not compared with ground truth |
| VSL fingerspelling (static letters, `signbridge/letters/`) | yes: recorder, trainer, live demo, Teams tab | yes (features, labels, checkpoint, typing, split; a synthetic 2-shape training check) | **no**: no letters recorded yet | **no**: no trained model, no accuracy. See §3. |
| CSLR (video → gloss sequence) | interface + untrained reference modules | yes | shapes only, on real batches | **no**: no training loop, no gloss data, no WER |
| SLT (→ Vietnamese text) | interface + untrained reference module | yes | shapes only, on real How2Sign English batches | **no**: no training, no Vietnamese data, no BLEU |
| TTS (Vietnamese text → speech) | **yes** (Piper, offline) | yes | yes: 8 hand-written Vietnamese sentences synthesized | **partly**: audio is produced, non-silent and finite; perceptual quality not rated |
| **ASL greetings demo** (29 isolated signs → phrase table → speech) | yes (`signbridge/signs/`) | yes | yes: ASL Citizen, 11 held-out signers | **yes, offline only**: test top-1 83.1 % (mean of 3 seeds), word-level 88.4 %. End-to-end on scripted test conversations with the current live defaults (IMAGE mode, seed 0, rest line 1.1): WER 25.0 %, 53/90 sentences exact. Live accuracy with a webcam user is **not measured**. See [asl_greetings_demo.md](asl_greetings_demo.md) §5–§6b. |
| WER metric (PHOENIX) | yes (`signbridge/metrics/`) | yes | yes: reproduces the official sclite output exactly | n/a (metric) |
| End-to-end **continuous** webcam → speech | **no** | — | — | **no** |

## 2. Component detail

### 2.1 Genuinely implemented and tested with real data

| Component | File | Real-data evidence |
|---|---|---|
| Common manifest schema, JSONL/CSV I/O | `signbridge/data/schema.py` | 1,741 How2Sign rows × 3 views; reproducible SHA-256 |
| How2Sign adapter (3 views) | `signbridge/data/how2sign.py` | official val CSVs, including the 8 quoted sentences |
| Video reader (seek/grab), image-folder reader, probe | `signbridge/video/io.py` | H.264 at 24/30/60 fps; seek frame-exact vs grab on 1 real video |
| Sampling, windows, resize/normalize, missing-frame policy, streaming | `signbridge/video/` | 34 clips; 46 s video streamed at 31.8 MB peak |
| Video ↔ annotation validation, clip localization | `signbridge/data/validation.py` | 11/11 clips located at original timestamps (±1 frame) |
| Variable-length collate, text vocabulary | `signbridge/data/collate.py`, `signbridge/text/vocab.py` | real How2Sign sentences |
| Vietnamese TTS | `signbridge/tts/` | Piper voice `vi_VN-vais1000-medium`, 8 sentences, 8 real-voice tests |

### 2.2 Tested only with synthetic data

| Component | Why only synthetic |
|---|---|
| VSL `continuous_csv` adapter (proposed layout) | no continuous VSL recordings exist yet |
| VSL `letter_frames` adapter (image folders) | no VSL letter image folders are part of the project |
| VSL fingerspelling module (`signbridge/letters/`) | no letters recorded yet |
| Missing/corrupt-frame policies (`zero`, `drop`, `CorruptVideoError`) | the real clips had no corrupt frames |
| `ManifestDataset` from an external JSONL | no external annotation export exists yet |
| CTC greedy collapse rule | checked on a hand-built probability path; there are no trained outputs |

### 2.3 Implemented as interfaces only (no trained weights, outputs meaningless)

| Component | File | What was actually checked |
|---|---|---|
| `FrameCNNEncoder`, `TorchvisionFrameEncoder` (random init), `PoseMLPEncoder` | `signbridge/models/reference.py` | shapes; padding does not leak into features |
| `TemporalConvEncoder`, `TransformerTemporalEncoder` | same | shapes; downsampled lengths |
| `CTCGlossDecoder` | same | finite CTC loss + backward on synthetic targets (tests); on real labels only when `validate_pipeline.py --vsl-root` is given |
| `TransformerTranslationModel` | same | teacher-forcing logits shape; generation loop terminates |
| `SignTranslationPipeline` | `signbridge/models/interfaces.py` | the gloss-free and gloss modes wire together |

### 2.4 Not implemented

* **Training:** training loop, optimizer and LR schedule, checkpoint save/resume, experiment
  config and logging, train-time augmentation.
* **Metrics:** BLEU / chrF / ROUGE (SLT) and their evaluation scripts. WER exists
  (`signbridge/metrics/`), but no CSLR model output has been scored with it.
* **CSLR decoding:** CTC beam search, and stitching windowed CTC outputs for continuous or
  live inference.
* **Feasibility check:** a guard for infeasible CTC samples. `nn.CTCLoss(zero_infinity=True)`
  currently zeroes them silently; see [cslr_data_requirements.md §4](cslr_data_requirements.md).
* **Pretrained weights:** loading for the video backbones (default `weights=None`).
* **Data:** adapters for any gloss-annotated dataset (PHOENIX, CSL-Daily, …).
* **Translation:** gloss-to-text translation model and its data adapter.
* **Live input:** webcam `FrameSource` for the CSLT path.
* **Integration:** end-to-end application wiring (webcam → CSLR → SLT → TTS).

## 3. VSL fingerspelling: not yet trained

Design of `signbridge/letters/`:

* MediaPipe hand landmarks → 83 hand-shape features (position-, size- and handedness-invariant)
  → small MLP.
* Data is recorded with `scripts/record_letters.py`, which stores landmarks only, never images.
* `scripts/train_letter_classifier.py` holds out whole takes. With a single signer it reports
  how well the model recognises **that signer**, not a new person. `--test-signer` holds out
  a whole signer.

**Status on 2026-11-23:**
* No letters have been recorded, so no model exists and **no accuracy has been measured**.
* The live demo and the Teams tab refuse to recognise without a trained checkpoint.
* The only evidence so far:
  * unit tests on synthetic landmarks;
  * a smoke run of the trainer on synthetic takes. It proves the code runs, not that the
    model recognises VSL.

## 4. What may and may not be claimed today

| Claim | Allowed? |
|---|---|
| "Video ingestion and preprocessing work on real sign language video" | Yes |
| "The architecture supports CSLR and SLT" | Only as "the interfaces exist and are shape-checked" |
| "SignBridge recognizes continuous signing" | **No** |
| "SignBridge translates sign language into Vietnamese" | **No** |
| "SignBridge speaks Vietnamese text aloud" | Yes, for supplied text (offline Piper voice). Quality not yet rated. |
| Any WER / BLEU / accuracy for CSLR or SLT | **No.** None has been measured. |
| "The demo recognizes 29 ASL greeting signs from unseen signers" | Yes: about 83 % per sign and 88 % per word on 11 held-out ASL Citizen signers. Live accuracy with a real webcam user has not been measured. |
| "The demo translates sign language" | **No.** Say "recognizes signs one at a time and composes a sentence from a fixed phrase table". |
| "SignBridge recognizes the VSL alphabet" | **No, not yet.** The letter module exists but has no trained model or measured accuracy. |
