# SignBridge Pipeline Validation Report (How2Sign subset)

**Date:** 2026-11-23. **Scope:** infrastructure validation only. **No model was trained.**
**No accuracy, WER or BLEU is reported**, because none was measured.

**Environment** (recorded by the script): Python 3.14.2, Windows 11 (10.0.26200), NumPy 2.4.6,
OpenCV 5.1.0, PyTorch 2.15.0+cpu (16 threads), MediaPipe 1.2.3. CPU only.

**Data:** the How2Sign val subset described in [dataset_audit.md](dataset_audit.md) (34 clips,
1 full video, 1,741 annotation rows). The validation experiment (§2) was re-run on 2026-11-23 at
21:05 without any VSL input. The VSL checks run only when `--vsl-root` points to a
`letter_frames` folder, and no such folder is part of the project.

## 1. Automated tests (`python -m pytest`)

**62 passed, 0 failed** in 13.4 s (the four infrastructure test files, re-run 2026-11-23):

* 55 synthetic tests need no dataset.
* 7 real-data tests are marked `realdata` and skip automatically when the subset is absent.
* `pytest -m "not realdata"` → 55 passed.

Counts below include parametrized cases.

| File | Tests | Covers |
|---|---|---|
| `tests/test_video.py` | 24 | probing; missing files; index-ordered streaming reads; time-cut segments in both seek modes; out-of-range segments; image folders; all sampling strategies; windows; letterbox resize; normalization; `max_frames`; the 3 missing-frame policies; corrupt-video rejection; streamed windows equal whole-clip windows; JSON config round-trip; 24 vs 30 fps time alignment |
| `tests/test_data.py` | 19 | schema nulls and validation; status inference; JSONL/CSV round-trip; How2Sign adapter (3 views, CSV quoting, missing clip, wrong columns, missing CSV); VSL `letter_frames` + proposed `continuous_csv`; one code path for both datasets; Vietnamese tokenization; empty-gloss refusal; padding; collate with missing gloss; `validate_sample`; clip localization |
| `tests/test_models.py` | 12 | shape validation; encoder padding invariance; torchvision/pose encoders; temporal downsampling lengths; transformer encoder; CTC loss and backward; greedy CTC collapse rule; translation teacher forcing and generation; pipeline modes |
| `tests/test_realdata.py` | 7 | 1,741 rows / 132 videos / no gloss; quoted sentences decoded; real clips decode with measured fps; clips match **original** not re-aligned timestamps; re-aligned segments inside video; container seek frame-exact on real H.264; real-clip preprocessing |

## 2. Validation experiment (`python scripts/validate_pipeline.py`)

Result: **10 PASS, 1 EXPECTED_REFUSAL, 0 FAIL, 0 SKIP**. The full output is in
`outputs/validation/results_no_vsl.json`. Times are wall-clock on the machine above, with other
jobs running, so they vary between runs.

| # | Check | Status | Time | Measured result |
|---|---|---|---|---|
| 1 | Manifest generation (3 How2Sign views) | PASS | 0.99 s | 1,741 records per How2Sign view (34 / 11 / 11 with local video). Gloss is null for all 1,741 rows; signer is null everywhere. 0 build warnings. |
| 2 | Video decoding + video ↔ annotation validation (full decode) | PASS | 16.5 s | 34 clips: 4,554 frames decoded at **280 frames/s**, 0 errors. 11 re-aligned segments valid. 7 duration warnings, all on suffix-`5` clips (+0.58 to +3.35 s). |
| 3 | Frame sampling + resize/normalize (default config: 12 fps, 224², letterbox) | PASS | 17.0 s | 34 clips → 1,970 steps. **Per clip: min 80 ms, median 314 ms, max 1,660 ms.** Source fps 24/30/60 → 11.2–12.0 steps per second of video. Output (T, 3, 224, 224) float32. |
| 4 | Window streaming on the 46 s full video | PASS | 9.5 s | 552 steps → 68 windows (16 / stride 8). Streaming 4.1 s with **31.8 MB peak**; whole-clip 5.3 s with **2,191 MB peak**. Same window count both ways. |
| 5 | Temporal alignment: clips inside the full video | PASS | 22.9 s | 11/11 clips found at the **original** start, offset 0–1 frame (criterion: \|offset\| ≤ 1). MAD 5.45–5.57 there, vs 8.6–14.1 at the re-aligned start. |
| 6 | Pose features: MediaPipe 2-hand keypoints (4 clips, 12 fps) | PASS | 14.4 s | **≈ 57–63 ms per frame.** Per-clip hand-slot detection ratios 0.796–1.00. Frames with no hand detected: 0–3.9 %. Output per step: (42, 4). |
| 7 | Variable-length batching + text vocabulary | PASS | 0.25 s | Batch (8, 148, 3, 224, 224), lengths 9–148, padding ratio 0.61. Text ids (8, 42). Vocabulary of 261 tokens, from these 34 sentences only. |
| 8 | Gloss supervision on How2Sign | EXPECTED_REFUSAL | 0.0 s | `GlossVocabulary` refuses: "no gloss annotations found…". `collate(require_gloss=True)` raises `AnnotationMissingError`. This is the correct behavior. |
| 9 | Model I/O contracts on real batches (**untrained**) | PASS | 2.2 s | RGB gloss-free path: encoder (8, 74, 64) with halved lengths; logits (8, 41, 261); forward 2.1 s on CPU; generation loop terminates. Pose encoder (4, 103, 32). The CTC contract on real labels was not run (no `--vsl-root`); it is covered on synthetic targets by `tests/test_models.py`. |
| 10 | Error handling | PASS | 0.03 s | Missing video → `VideoUnavailableError`; missing gloss → `AnnotationMissingError`; unknown id → `SampleNotFoundError`. The 2 sentences absent from the official archive are flagged `video_available=false`. |
| 11 | Reproducibility | PASS | 2.6 s | Identical SHA-256 for two manifest builds, two preprocessing runs, and two seeded model forwards. Subset selection is recorded (seed 0). |

The audit script (`scripts/audit_how2sign.py`) and the manifest builder
(`scripts/build_manifests.py`) also completed without errors (re-run 2026-11-23).

## 3. Was gloss recognition genuinely validated?

**No.** How2Sign's public release has no gloss annotations (see audit §1 and §4). The following
were validated:

* the CTC decoder's **tensor contract**: shapes, lengths, a finite loss and backward pass, and the
  greedy collapse rule on a hand-built path;
* that the pipeline **refuses** to build gloss supervision when none exists.

No CSLR model was trained, and no WER exists. CSLR stays unvalidated until a dataset with real
gloss sequences is available, ideally VSL with VSL glosses.

## 4. Was translation genuinely validated?

**No, only the data side and the interface.**

* **Validated:** 1,741 real sentence ↔ English pairs load, tokenize and batch correctly. An
  untrained translation model runs teacher forcing and generation on real batches with correct
  shapes.
* **Not validated:** translation quality. No model was trained, and outputs of the random-weight
  model are meaningless, so they are not reported. No BLEU exists.

## 5. Known limitations

* **Small subset:**
  * 34 of 1,739 val clips (2 %) and 1 of 132 full videos;
  * the temporal alignment evidence comes from one video (11 sentences);
  * train and test splits were not touched.
* **Suffix-`5` clip durations** differ from their original interval by up to 3.35 s (7 of 8
  sampled). The cause is unknown, so those clips' exact boundaries are untrusted.
* **Re-aligned timestamps** apply only to full videos (17.5 GB for val). The adapter supports this
  (`realigned_full` view), but at scale it requires downloading full videos. Whether re-aligned
  boundaries match the signing better was not measured, because there is no sign-level ground truth.
* **OpenPose keypoint schema not verified.** Only the archive head was read, and it contained
  rendered videos. The pose path was validated with MediaPipe instead.
* **Handedness slots** use MediaPipe's handedness labels as reported. Their correctness on
  mirrored or non-mirrored footage was not verified.
* **Preprocessing throughput** is CPU-bound: about 231 ms median per clip at 12 fps, 224² (single
  process).
* **Seeking.** The default `seek` mode relies on OpenCV/FFmpeg container seeking. On the real
  H.264 full video it returned frames byte-identical to frame-exact `grab` mode for 3 segments
  (87–188 frames), and was 1.1–3.5× faster. This was checked on one video only, so `grab` remains
  available when exactness matters more than speed.
* **Signer ids are null**, so signer-independent splits cannot be built from How2Sign metadata here.
* **The continuous VSL layout is a proposal**, tested only on synthetic fixtures.
* **Reference models are random-init.** Their shapes are meaningful; their outputs are not.

## 6. Acceptance criteria

| Criterion | Status |
|---|---|
| How2Sign data can be inspected and loaded | Met (val subset) |
| Dataset manifests are generated correctly | Met (nulls for missing fields, tested) |
| Video preprocessing works | Met (real 24/30/60 fps H.264 clips) |
| Temporal sequences can be constructed | Met (fps resampling, windows, streaming) |
| Annotation alignment validated where data supports it | Met for clip ↔ video ↔ original timestamps (1 video); re-aligned only checked for in-bounds |
| Dataset-specific code separated from model architecture | Met (`signbridge/data/{how2sign,vsl}.py` only) |
| VSL adapter can be implemented without rewriting the architecture | Met at interface level (`letter_frames` and `continuous_csv` on synthetic fixtures) |
| Automated tests pass | Met (62/62) |
| Documentation accurately describes limitations | See §3–§5 |
