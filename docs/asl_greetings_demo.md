# Live demo: ASL greetings → English / Vietnamese sentence → speech

**Date:** 2026-11-23. **Status:** implemented, trained, and evaluated on held-out signers.
Tested end-to-end on video and smoke-tested on the webcam.

## 1. What it is, and what it is not

| It is | It is not |
|---|---|
| Recognition of **29 isolated ASL signs** for greetings and introductions, one sign at a time, hands lowered between signs | Continuous sign language recognition (no co-articulation, no fingerspelling, no names) |
| A **hand-written phrase table** that maps the recognized sign sequence to an English sentence and a Vietnamese equivalent | A learned translation model. Unknown sequences are shown as `[gloss]` words. |
| Speech: English through the Windows SAPI voices (offline), Vietnamese through Piper | VSL. This demo is ASL, because no usable public VSL data exists (see `cslr_data_requirements.md`). |

**Vocabulary (29 classes).** The numbered entries are lexical variants that map to the same
word:

HELLO, NAME, MY, YOUR, YOU, ME, NICE, MEET, WHAT1, WHAT2, HOW1, HOW2, FINE1, FINE2, GOOD,
THANKYOU, SORRY, PLEASE, BYE, YES, NO, DEAF1, DEAF2, HEARING, LEARN, SIGN, FRIEND, SEE, LATER.

## 2. Data: ASL Citizen subset

* **Source:** ASL Citizen (Microsoft Research).
  * 52 Deaf / hard-of-hearing signers recording on home webcams.
  * Official signer-independent splits.
  * License: **Microsoft Research License Terms**: non-commercial research only, no
    redistribution, personal data to be deleted when the research ends. Accepted by the
    project owner on 2026-11-23.
* **Selective download, no full download.** `scripts/fetch_asl_citizen.py` opens the
  42.8 GB zip remotely through HTTP byte ranges and fetches only the selected clips (CRC
  checked by `zipfile`).
  * Index and split CSVs: 8 requests, 13.8 MB.
  * The 29 signs: **895 clips, 966 MB**, in 612 s.
* **Storage:** `../datasets/asl_citizen/`, outside Git. Do not publish clips or frames.

| Split | Clips | Signers |
|---|---|---|
| train | 436 | 30 |
| val | 105 | 5 |
| test | 354 | 11 |

* Train / test signer overlap: **none** (asserted in the training script).
* About 15 training clips per sign, which is small.

## 3. Features and model

* **Landmarks** (`signbridge/signs/features.py`): MediaPipe PoseLandmarker lite and
  HandLandmarker (2 hands) per frame.
  * Hands are matched to the person's left/right by the nearest pose wrist.
  * The same code runs offline and live.
* **Extraction** (`scripts/extract_sign_features.py`): 6 processes, 1,602 s for 895 clips.
  * Every clip had a detected hand.
  * Pose detection rate ≥ 98.3 % per clip.
  * A first run with 16 processes broke the process pool (memory), so the script is now
    resumable and rebuilds the pool after a crash.
* **Feature vector:** 348 values per frame, at T = 32 frames resampled from the active span
  (the frames with a visible hand).
  * coordinates relative to the shoulder midpoint, in units of shoulder width;
  * hand shape relative to the wrist;
  * hand presence;
  * first differences (motion).
* **Model** (`signbridge/signs/model.py`):
  * LayerNorm → Linear → 2-layer BiGRU(128) → attention pooling → 29 classes.
  * Training: AdamW with a one-cycle LR schedule, 150 epochs on CPU (about 7 min).
  * Augmentation: mirror (left-handed signers), small rotation, scale and shift, temporal
    crop, frame dropout.
  * The checkpoint is chosen on **val**; **test** is evaluated once.
  * Checkpoint: `../models/signs/asl_greetings.pt` (outside Git).

## 4. Results: held-out test signers (11 signers, 354 clips)

| Seed | Val top-1 | **Test top-1 (29 signs)** | Test top-3 | **Test word-level top-1** |
|---|---|---|---|---|
| 0 (demo checkpoint) | 92.4 % | **83.9 %** | 95.8 % | **89.3 %** |
| 1 | 93.3 % | 83.3 % | 97.5 % | 89.0 % |
| 2 | 92.4 % | 82.2 % | 97.7 % | 87.0 % |
| mean | 92.7 % | **83.1 %** | 97.0 % | **88.4 %** |

* *Word-level* counts lexical variants (FINE1/FINE2 …) as the same word. That is what the
  demo shows, because the phrase table merges them.
  Computed by `scripts/eval_sign_classifier.py`.
* **Confidence threshold 0.45** (seed 0):
  * 93.8 % of signs are accepted;
  * 91.9 % of accepted signs are the right word;
  * a wrong word is shown for 7.6 % of signs.
* **Per signer** (word level): 76.5 % to 100 %.
* **Main word confusions:** GOOD↔THANK-YOU (10; both start at the chin), FINE→YOUR (3),
  ME→SORRY (3), PLEASE↔MY (4), YOU↔DEAF (3).

## 5. End-to-end test of the live pipeline (held-out signers)

* **Videos** (`scripts/make_demo_test_video.py`):
  * For 10 test signers, the clips of a scripted 9-sentence conversation (18 signs) are
    concatenated with 3 s pauses.
  * Script: HELLO / NICE MEET YOU / HOW YOU / ME FINE / YOUR NAME WHAT / ME DEAF / THANK-YOU /
    SEE YOU LATER / BYE.
  * The videos stay local (license).
* **Run:** `scripts/live_sign_demo.py --video … --headless`. Exactly the live code, with
  timestamps from the video fps.
* **Scoring:** `scripts/score_demo_test.py`.

| Metric | Result |
|---|---|
| Signs performed / segments detected | 180 / 183 |
| Word error rate (accepted signs vs. script, variants merged) | **24.4 %** (S 22, D 18, I 4) |
| Sentences reproduced exactly (English) | **54 / 90 (60 %)** |
| Best / worst signer (WER) | 11.1 % (P47) / 44.4 % (P22) |

Why the end-to-end result is lower than the clip accuracy:

1. **Systematic errors for some signers.** P22's YOU is read as DEAF with 90 % confidence in
   all 3 occurrences, and in a sentence one wrong word breaks the whole phrase.
2. **Rejected signs** (below the threshold) become deletions.
3. **Segmentation.**
   * P15: a sign split in two, causing 4 insertions.
   * P49: very short clips, and the "rest" frames still show a hand at the frame edge. After
     video re-encoding MediaPipe detects it in about every other frame, so the pause is not
     recognized and sentences merge (5 output sentences instead of 9).
   * The same risk exists live when a hand stays partly in view. Lower the hands fully out of
     the frame.

**Harness fixes.** These numbers are after two fixes to the test harness, not to the model or
the demo settings:

* First version: the pause was a frozen last frame, sometimes with the hands still up. Result:
  WER 25.0 %, 54/90.
* Second version: the pause is the idle frame farthest from any hand activity. Result:
  WER 24.4 %, 54/90.
* The live thresholds were **not** tuned on these test videos.

## 6. Webcam smoke test (this laptop, CPU)

* 10 s headless: **151 frames, 30.7 ms processing per frame**. The pipeline could run at about
  32 fps; the camera delivered about 15 fps.
* The UI (skeleton, status, EN/VI lines with Vietnamese diacritics) was rendered and checked
  on a dataset frame. That image was then deleted, because it shows a dataset participant.
* **The 30.7 ms figure did not hold in a real windowed session (2026-11-23).** The user reported
  lag. A per-stage profile on the webcam, with the window open and the CPU about 48 % busy with
  other applications, measured a loop of only **7.75 fps**:

  | Stage | ms / frame |
  |---|---|
  | `cap.read()` | 0.2. The frame was already waiting in the driver buffer, so it was **stale**. That is the visible lag. |
  | MediaPipe pose | 34.6 |
  | MediaPipe hands | 32.0 |
  | UI text (PIL) | **49.0**. The font was reloaded and the full frame converted twice, every frame. |

  The camera alone delivers 29 fps.
* **Fixes:**
  * a capture thread that keeps only the newest frame;
  * pose and hands in parallel. Same output; 74.6 → 48.6 ms in a bench;
  * the text panel is cached and only re-rendered when its text changes;
  * the speech voice is loaded at start-up.
  * MediaPipe `--mp-mode video` (tracking across frames) is now the **live default**. It took
    37.5 ms in the bench.
    * Its landmarks differ slightly from the IMAGE-mode landmarks used in training, so it was
      checked on **val** before being chosen. Val top-1 was the same: 97/105 (92.4 %) in both
      modes, with 4 clips changing prediction.
    * Re-run afterwards on the §5 end-to-end test videos, for information only. These were
      not used for the choice.

      | Mode | WER | S/D/I | Sentences exact |
      |---|---|---|---|
      | image (§5) | 24.4 % | 22/18/4 | 54/90 |
      | video (default) | **20.0 %** | 17/15/4 | **59/90** |

      With 180 signs this difference is small and may be noise. P49's merged sentences
      (5 out of 9) did not change.
    * `--mp-mode image` reproduces the training extraction exactly.
* **After the fixes,** a 10 s windowed webcam run, with nobody signing and the same background
  load, measured **18.9 fps** (was 7.75): 48.8 ms per frame for landmarks and model, about 4 ms for
  the UI.
* **User feedback after the fixes:** less lag, but recognition still not fast. The user asked to
  **prioritize quality**. See §6a.
* **Not done:** a person signing live in front of the webcam.

## 6a. Quality pass (2026-11-23): choices made on val, test run once

**Tools.**
* `scripts/make_demo_test_video.py --split val` builds the §5 conversation from the
  **validation** signers. Only 3 of the 5 have all 15 signs: 27 sentences, 54 signs.
* `scripts/tune_live_config.py` caches the landmarks of each conversation once (MediaPipe VIDEO
  mode, as live) and replays `SignSession` with different settings.
* Check: replaying the single seed-0 model on the test conversations gives exactly the live
  `--video` run (WER 20.0 %, S/D/I 17/15/4, 59/90).

**1. Averaging the 3 training seeds (now the default `--ckpt`).**

| | Val clips (105) | Val conversations (54 signs) | Test clips (354) | Test conversations (180 signs) |
|---|---|---|---|---|
| seed 0 alone | 92.4 % sign / 94.3 % word | WER 7.4 %, 18/27 | 83.9 % / 89.3 % | WER **20.0 %** (17/15/4), **59/90** |
| seed 1 / seed 2 | 93.3 / 92.4 % sign | WER 11.1 / 13.0 % | 83.3 / 82.2 % | not run |
| **3-seed average** | **94.3 % / 97.1 %** | WER 7.4 %, 18/27 | **86.2 % / 91.2 %** | WER 21.1 % (25/9/4), 54/90 |

* It was chosen on val: best on clips, and tied with the best single seed on conversations. A
  real deployment would not know in advance which seed is best.
* On test it is **better per clip** (+2.3 points per sign, +1.9 per word, with the same 7.6 % of
  wrong words shown at threshold 0.45).
* It is **slightly worse on the test conversations.** It accepts more segments (deletions
  15 → 9) but substitutes more (17 → 25).
* The conversations reuse 180 of the 354 test clips, so the clip result is the larger
  sample. The trade-off is reported, not hidden.
* `--ckpt ../models/signs/asl_greetings.pt` runs the single model.
* Cost: 3 small GRU passes once per sign, which is negligible next to MediaPipe.

**2. Rejected on val.**
* **Mirror test-time augmentation:** val 94.3 → 93.3 % with the ensemble, 92.4 → 89.5 %
  alone.
* **Debouncing hand activity** (`LiveConfig.active_window/active_min`, 2 of 3 or 3 of 5
  frames): val conversation WER 7.4 → 14.8 / 11.1 %, because more signs were missed. The option
  is kept, off by default.
* **`end_gap_s` 0.3 / 0.6** instead of 0.4: no gain (7.4 / 9.3 %).
* **Threshold 0.6** instead of 0.45: on val clips the wrong words shown fell 1.9 → 1.0 %, but
  accepted signs fell 97.1 → 90.5 %. Kept at 0.45.

## 6b. First live session by the user (2026-11-23): what went wrong, and what changed

**The user's report.** The §6a version (VIDEO mode, 3-seed average) recognized their signs
worse than the first version did.
* There is **no log of the first-version session**, so the comparison cannot be measured.
* The §6a session log (`outputs/signs/live_session1.json`, 27 segments) shows:
  * **14 of 27 segments were cut at `max_sign_s` = 4 s.** The hands were never seen as
    resting, so one segment mixed several signs and the rest pose.
  * Confidence was mostly 0.14–0.35. Only 11 of 27 segments were accepted.
* **Root cause.** The rest line was 1.6 shoulder widths below the shoulders, about hip level.
  * In the user's framing that is image y ≈ 0.94.
  * Hands resting on the desk sat at y ≈ 0.89 (1.27 shoulder widths below the shoulders),
    **above** the line.
  * So 98 % of the frames counted as "signing" in an 8 s check with the hands on the desk.
  * ASL Citizen signers lower their hands out of view, so no offline evaluation could reveal
    this.
  * VIDEO-mode tracking keeps a resting hand "present" more consistently, which plausibly
    made it worse than the first version.

**Changes.**
* **Recognition defaults back to the first version:** `--mp-mode image` and the single seed-0
  model. The lag fixes from §6 stay.
* **`rest_line` 1.6 → 1.1** (`--rest-line`).
  * In train+val, the highest wrist of 99.6 % of sign cores (the middle 60 % of the active
    span) is above 1.1. The lowest glosses are YOUR at 0.80 and SORRY at 0.62.
  * The val conversations are unchanged at rest lines 1.6, 1.2, 1.1 and 1.0: WER 7.4 % and
    18/27 in each case.
  * Test conversations, run once (IMAGE mode, seed 0):
    * at 1.6 the replay reproduces §5 exactly: 24.4 %, 22/18/4, 54/90;
    * at **1.1: 25.0 %, 22/19/4, 53/90**. One more sign was missed out of 180.
  * This small offline cost is accepted, because at 1.6 a user whose hands rest on a desk
    never ends a sign.
* **On screen:**
  * the rest line is drawn;
  * a warning appears when a sign is cut at 4 s.
* **For the next session:**
  * the log records `ended_by` (`rest` / `max` / `flush`) for each sign;
  * `--save-landmarks` stores the per-frame landmarks (no images), so a session can be
    diagnosed and replayed.

**Why the demo is not "fast".** A sign is classified only after both wrists have been below the
rest line (or out of view) for `end_gap_s` = 0.4 s. A sentence is composed after a 2 s pause
(SPACE ends it at once). Shorter gaps were not better on val, so quality was kept over speed.

**No live accuracy has been measured.** Expect it to be lower than §5 for signers whose
signing, lighting or framing differ from ASL Citizen.

## 7. How to run

```powershell
python -X utf8 scripts/live_sign_demo.py                 # webcam, English speech (Windows voice)
python -X utf8 scripts/live_sign_demo.py --speak vi      # Vietnamese speech (Piper)
python -X utf8 scripts/live_sign_demo.py --speak off --min-confidence 0.6
python -X utf8 scripts/live_sign_demo.py --mp-mode video  # faster tracking; worse when hands rest in view (§6b)
python -X utf8 scripts/live_sign_demo.py --log outputs/signs/live_session2.json --save-landmarks outputs/signs/live_session2.npz
```

The footer shows the loop rate in fps and the landmark time in ms. If the fps stays below about
12, close other heavy applications. A sign then has too few frames.

**Tips for a good demo:**
* Sit about 1 m from the camera, with your upper body and both hands in view while signing.
* Sign one sign at a time, then lower your hands **out of the frame**.
* Pause about 2 s to end the sentence, or press SPACE.

**Keys:** Q quit, SPACE end sentence, C clear, L switch speech language, M mute.

**Sentences that compose cleanly:** "HELLO", "NICE MEET YOU", "HOW YOU", "ME FINE",
"YOUR NAME WHAT", "ME DEAF", "ME LEARN SIGN", "THANK-YOU", "SEE YOU LATER", "BYE".

## 8. Licenses

* **ASL Citizen data and the model trained on it:** non-commercial research only. Do not
  publish the clips. The checkpoint is a derived work, so keep it to research use.
* **MediaPipe models:** Apache-2.0.
* **Windows SAPI voices:** part of the OS.
* **Piper:** GPL-3.0 (see `tts_implementation.md`).

## 9. Next steps

1. **Live check with a real signer:** record 2–3 sessions and log them with `--log`.
2. **GOOD vs THANK-YOU:** add more training data (more ASL Citizen signers are not available
   for these glosses, so record your own clips), or drop GOOD from the phrase table.
3. **Segmentation:** debounce hand detection (a frame counts as active only when ≥ 2 of the
   last 3 frames are) and evaluate on **new** videos, not these test videos.
4. **VSL:** the same pipeline works unchanged with VSL clips. Record ~20 takes per sign from
   several signers, or use VieSign-10K.
