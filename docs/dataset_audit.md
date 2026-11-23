# How2Sign Dataset Audit (validation subset)

**Purpose.** Use How2Sign (ASL ↔ English) only to validate SignBridge's data pipeline before
adapting it to Vietnamese Sign Language. ASL is not VSL. No ASL label, gloss or translation is
mapped to Vietnamese.

**Audit date:** 2026-11-23. Every number below is produced by
[`scripts/audit_how2sign.py`](../scripts/audit_how2sign.py), which writes
`outputs/audit/how2sign_audit.json` (git-ignored). Unless stated otherwise, numbers refer to the
**val** split.

**License:** How2Sign is distributed under **CC BY-NC 4.0**: attribution required,
**non-commercial use only**. SignBridge may use it only for research validation. Do not ship it,
or models trained on it, in a commercial product without separate permission. Cite
Duarte et al., *How2Sign*, CVPR 2021.

---

## 1. Sources and resource availability

| Source | Result |
|---|---|
| Download script `how2sign/how2sign-data/download_how2sign.sh` | Cloned and read. Its Google Drive IDs for **val clips** (`1oVZ…`) and **val 2D keypoints** (`1aOh…`) return **HTTP 404**. Its annotation CSV IDs still work. The script ends with `## TODO # Gloss annotations`. |
| Website `how2sign.github.io` download table | Current, working Drive links (used by [`scripts/fetch_how2sign_subset.py`](../scripts/fetch_how2sign_subset.py)). |
| **Gloss annotations** on the website | The "Gloss Annotation" block is **inside an HTML comment** (not rendered). Its three placeholder links point to an unrelated OpenImages bounding-box CSV. **No gloss annotation is publicly downloadable.** Nothing was downloaded from those links. |
| "Metadata" block (categories, timestamps, statistics, camera parameters) | Also commented out in the HTML: not published. |
| Side-view 2D keypoints | Commented out: not published. |

Archive sizes as reported by the server: val frontal clips ZIP **1.78 GB** (1,739 members), val
frontal full-video ZIP **17.53 GB** (132 members), val 2D keypoints TAR.GZ **1.24 GB**.

## 2. What was actually downloaded

Whole archives were **not** downloaded. Google Drive honors HTTP `Range` requests, so the
fetcher reads each ZIP's central directory, then downloads only the byte ranges of selected
members and verifies every member's CRC32.

| Item | Files | Size on disk | Selection |
|---|---|---|---|
| `how2sign_val.csv` (original timestamps) | 1 | 311,131 B | whole file |
| `how2sign_realigned_val.csv` (manual re-alignment) | 1 | 311,419 B | whole file |
| Val frontal **sentence clips** | 34 | ≈ 28 MB | 24 random (`--seed 0`, ≤ 15 MB each) + all 11 clips of video `a4uz0W33REs-3-rgb_front` (one overlapped) |
| Val frontal **full video** | 1 | 57.3 MB | `a4uz0W33REs-3-rgb_front` (smallest val video), to test temporal segmentation |
| Archive listings (`_listings/*.json`) | 2 | small | central directories of the two ZIPs |
| Keypoints archive | 0 | none | Only the first 8 MB of the stream was read (see §7). |

Total on disk: 41 files, 86.4 MB, stored in `../datasets/How2Sign/`, **outside the
repository**. `.gitignore` also excludes `datasets/`, `outputs/`, `*.mp4`, `*.zip` and `*.tar.gz`.
Each fetch is logged with its CRC or SHA-256 in `_provenance.jsonl`.

## 3. Directory structure (official layout, reproduced locally)

```
How2Sign/
├── sentence_level/val/
│   ├── text/en/raw_text/how2sign_val.csv
│   ├── text/en/raw_text/re_aligned/how2sign_realigned_val.csv
│   └── rgb_front/raw_videos/<SENTENCE_NAME>.mp4        # ZIP member prefix: raw_videos/
└── video_level/val/rgb_front/raw_videos/<VIDEO_NAME>.mp4
```

## 4. Annotation schema

Both CSVs are **TAB-separated with standard CSV quoting**. They have one header row and seven
columns. No cells are empty.

| Column | Type | Example | Meaning |
|---|---|---|---|
| `VIDEO_ID` | str | `-d5dN54tH2E` | YouTube id of the How2 source video |
| `VIDEO_NAME` | str | `-d5dN54tH2E-1-rgb_front` | full How2Sign recording = `<VIDEO_ID>-<n>-rgb_front` |
| `SENTENCE_ID` | str | `-d5dN54tH2E_0` | `<VIDEO_ID>_<k>`. Some ids merge indices, e.g. `DfnHNkTE7mE_11_12`. |
| `SENTENCE_NAME` | str | `-d5dN54tH2E_0-1-rgb_front` | clip file stem, unique key |
| `START`, `END` *(original)* / `START_REALIGNED`, `END_REALIGNED` | float s | `0.44` / `12.97` | interval on the **full video** timeline |
| `SENTENCE` | str | "We're going to work on a arm drill…" | **English translation** (How2 speech transcript) |

Two findings from the audit:

* **Quoting.** 8 sentences are stored as quoted fields with doubled quotes
  (`"…an ""i"" you're…"`). A parser with quoting disabled corrupts them, so the adapter uses
  standard CSV quoting. A real-data test guards this.
* **No gloss column** exists in either file. `SENTENCE` is an English translation, **not** a gloss
  sequence, and is stored only as `target_text`.

The `SENTENCE` text is identical in both CSVs. Only the timestamps differ.

## 5. Counts and ID mapping

| Quantity | Value |
|---|---|
| Annotation rows (sentences) | **1,741** (same set of `SENTENCE_NAME`s in both CSVs) |
| Unique `VIDEO_ID` / `VIDEO_NAME` | 115 / **132** |
| `VIDEO_NAME` numeric suffix `n` | `1`: 276 rows, `2`: 270, `3`: 306, `5`: 454, `8`: 435 |
| Words per sentence (whitespace) | min 1, median 15, mean 17.5, max 103 |
| Lower-cased whitespace vocabulary | 4,418 types |
| Clips in archive | 1,739. **2 annotated sentences have no clip:** `eY32ru3Nstc_20-8-rgb_front`, `fE6xxSbjVV8_12-8-rgb_front`. |
| Clips without annotation | 0 |
| Full videos in archive ↔ annotated `VIDEO_NAME`s | 132 ↔ 132, no mismatch |

The suffix `n` probably identifies a recording session or signer, but nothing in the downloaded
files documents it. **`signer_id` is therefore left null.**

## 6. Video properties (all 34 clips + 1 full video, fully decoded)

| Property | Value |
|---|---|
| Container / codec | MP4 / H.264, all files |
| Resolution | 1280×720, all files |
| FPS (clips) | **24 fps: 31 clips, 30 fps: 1, 60 fps: 2** (mixed frame rates) |
| Clip frames | min 15, median 88.5, max 805 |
| Clip duration | min 0.63 s, median 3.10 s, max 17.29 s |
| Corrupted frames / reported ≠ decoded count | 0 / 0 |
| Full video `a4uz0W33REs-3` | 24 fps, 1,105 frames (all decodable), 46.04 s, 11 sentences |

Because frame rates are mixed, the preprocessing resamples to a common temporal rate
(`sampling="fps"`).

## 7. Modalities

| Modality | Status in this audit |
|---|---|
| RGB frontal sentence clips | Downloaded (subset), decoded, used |
| RGB frontal full videos | Downloaded (1 video), decoded, used for temporal alignment |
| RGB side view | Available upstream, not downloaded (not needed) |
| English translation (original and re-aligned) | Downloaded in full for val |
| B-F-H 2D keypoints (OpenPose) | **Not downloaded.** The 1.24 GB TAR.GZ cannot be read randomly. The first 8 MB contained only rendered `openpose_output/video/*.mp4` overlay videos, so the **keypoint JSON schema was not verified**. The pose path was validated with MediaPipe hand keypoints extracted locally instead. |
| Gloss annotations | **Not available** (see §1) |
| Speech audio / How2 data | Not redistributed by How2Sign (copyright) |

## 8. Temporal boundaries: original vs re-aligned

| Statistic (1,741 rows) | Original | Re-aligned |
|---|---|---|
| Segment duration | min 0.11, median 4.73, mean 5.67, max 53.22 s | min 0.02, median 5.70, mean 6.92, max 57.99 s |
| Non-positive intervals / negative starts | 0 / 0 | 0 / 0 |

The two timelines disagree substantially:

* IoU between the original and re-aligned interval: median **0.096**, mean 0.305.
* **809 of 1,741** rows have IoU = 0.
* Only 164 rows have the re-aligned interval inside the original one.
* The re-aligned start is later by a median of 3.22 s (range −5.85 to +63.74 s).

**Which timestamps were the released clips cut with?** The website says the original ones. We
checked this empirically by locating each clip's first 8 frames inside the full video
(`locate_clip_in_video`: 64×36 grayscale, mean absolute difference (MAD), ±2 s search).

* All **11/11** clips of `a4uz0W33REs-3` match the full video at the **original** `START`, within
  0–1 frame. MAD at the match is 5.45–5.57, which is the H.264 re-encode floor.
* At the **re-aligned** `START` the MAD is 8.6–14.1, so there is no match. In 2 cases the ±2 s
  search window reached back to the original start, and the best match landed there.
* Clip duration vs. original interval: |difference| median 0.023 s. Clips from videos with suffix
  `1, 2, 3, 8` agree within 0.042 s. **7 of 8 clips from suffix-`5` videos are 0.2–3.3 s longer
  than their original interval.** Cause unknown; they are flagged as warnings, not errors.
* All 11 re-aligned segments of the downloaded full video lie inside the video (46.04 s).

**Consequence.** Re-aligned boundaries cannot be applied to the released clips. To train on
re-aligned boundaries you must cut them from the **full videos** (17.5 GB for val). The adapter
exposes three views:

* `clips`: clip ↔ original row
* `original_full`: full video + original interval
* `realigned_full`: full video + re-aligned interval

## 9. Supported vs unsupported tasks (with this subset)

| Task | Supported? | Evidence / reason |
|---|---|---|
| **A. Video ingestion** | **Yes** | 34 clips and 1 full video decoded with no corrupted frames; fps 24/30/60 handled |
| **B. Temporal feature extraction** | **Yes (pipeline level)** | Sampling, windows, streaming, and MediaPipe 2-hand keypoints computed on real clips. RGB features come from *untrained* encoders: shape-valid only. |
| **C. Gloss sequence recognition (CSLR)** | **No** | No gloss annotations exist in the public release. **CSLR decoder training and evaluation cannot be validated on How2Sign.** Only the CTC interface's tensor contract was exercised, using real *VSL isolated letter* labels. |
| **D. Sentence-level translation (SLT)** | **Data: yes. Model: not validated.** | 1,741 sentence ↔ English pairs, loadable and batchable. No translation model was trained (out of scope), so translation quality is unknown. |
| **E. Temporal alignment** | **Partially** | Clip ↔ full-video ↔ original timestamps verified to ±1 frame (1 video, 11 sentences). Re-aligned segments are loadable from full videos, but their correctness vs. the signing cannot be verified without gloss or sign-level ground truth. |

## 10. Missing fields (normalized manifest)

| Field | How2Sign value | Why |
|---|---|---|
| `gloss_sequence` | `null` (all 1,741) | not published |
| `signer_id` | `null` | suffix meaning undocumented |
| `source_text` | `null` | English is the translation target, stored in `target_text` |
| `fps` | measured for the 34 local clips; `null` otherwise | never guessed |
| `start_time`/`end_time` | `null` in `clips` view (the whole file is the segment); set in the `*_full` views | — |
| `annotation_status` | `translation_only` | — |

## 11. Reproduce

```
python scripts/fetch_how2sign_subset.py annotations
python scripts/fetch_how2sign_subset.py fetch val_rgb_front_clips --n 24 --seed 0
python scripts/fetch_how2sign_subset.py fetch val_rgb_front_clips --names <11 a4uz0W33REs_*-3-rgb_front.mp4>
python scripts/fetch_how2sign_subset.py fetch val_rgb_front_videos --names a4uz0W33REs-3-rgb_front.mp4
python scripts/audit_how2sign.py
```
