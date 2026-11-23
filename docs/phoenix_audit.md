# RWTH-PHOENIX-Weather 2014-T: download and audit

**Date:** 2026-11-23. **Status:** downloaded (subset kept), WER metric verified against the
official scorer. No PHOENIX model has been trained yet (needs a GPU; see the roadmap).

**License:** CC BY-NC-SA 3.0, research only. The data lives in `../datasets/phoenix2014T/`,
outside Git.

## 1. Download

* **Source:** `https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz`.
  The server reported 41,699,758,035 bytes and supports byte ranges.
* **Script:** `scripts/fetch_phoenix.py`. It streams the `.tar.gz` and decompresses it on the
  fly, so the archive itself is never stored. Measured: 57.6 min at about 12 MB/s.
* **Why a subset.** The disk had 49.8 GB free, which is not enough to keep everything.
* **What is kept** (`fullFrame-210x260px` PNG frames):
  * annotations, evaluation scripts and the language models (all files outside `features/`);
  * all of **dev** (519 videos) and **test** (642 videos);
  * a reproducible **1/8 of train**: 911 of 7,096 videos, chosen by `md5(video name) % 8 == 0`.
* **Incident.** The first version of the script computed the archive root from the first tar
  member, which is the bare top-level directory. Every file was therefore classified as
  "other" and kept. The non-selected train folders were deleted during and after the run
  (32 GB freed), and the bug is fixed (the root is now found from a member that contains
  `PHOENIX-2014-T/`).
* **Inventory.** `inventory.json` was written by that buggy run, so per-member data is in
  `other_files`. The counts below come from it.

## 2. Contents (full release, counted from the stream)

| Split | Videos | Frames | Size |
|---|---|---|---|
| train | 7,096 | 827,300 | 36.25 GB |
| dev | 519 | 55,769 | 2.44 GB |
| test | 642 | 64,627 | 2.83 GB |
| outside `features/` | 26 files | — | 7.7 MB |

* Frames are 210×260 PNG, named `images0001.png` and up, at 25 fps (per the dataset page;
  the PNG folders carry no timing).
* Annotation rows match the video counts: train 7,096, dev 519, test 642
  (`annotations/manual/PHOENIX-2014-T.<split>.corpus.csv`).

## 3. Annotation format (inspected)

The files are pipe-separated with the header:

```
name|video|start|end|speaker|orth|translation
11August_2010_Wednesday_tagesschau-1|11August_2010_Wednesday_tagesschau-1/1/*.png|-1|-1|Signer08|JETZT WETTER MORGEN DONNERSTAG ZWOELF FEBRUAR|und nun die wettervorhersage für morgen donnerstag den zwölften august
```

| Column | Content |
|---|---|
| `orth` | DGS gloss sequence |
| `translation` | German sentence, lower-cased and tokenized |
| `speaker` | signer id |
| `start`, `end` | `-1`: whole clip |

* The `video` column has a `/1/` component that does not exist on disk. The real path is
  `features/fullFrame-210x260px/<split>/<name>/images*.png`.
* `train-complex-annotation.corpus.csv` keeps the meta tokens (`__ON__`, `__OFF__`, …).
* A `PhoenixDataset` adapter has **not** been written yet (the work moved to the demo).

## 4. WER metric checked against the official scorer

* The release ships the official script (`evaluation/sign-recognition/evaluatePHOENIX-2014-T-signrecognition.sh`,
  which uses grep/sed/awk/perl plus NIST sclite), an example hypothesis for dev and test, and
  sclite's output on it.
* `signbridge/metrics/phoenix.py` ports the normalization rule by rule.
  `signbridge/metrics/wer.py` aligns with sclite's weights: substitution 4, deletion 3,
  insertion 3.
* `scripts/check_phoenix_wer.py` compares the two. Nothing from the dataset is executed.

| Split | Reference words | Sub | Del | Ins | **WER** |
|---|---|---|---|---|---|
| dev, official sclite | 3,748 | 12.0 | 9.7 | 3.0 | **24.7** |
| dev, signbridge | 3,748 | 12.0 | 9.7 | 3.0 | **24.7** |
| test, official sclite | 4,264 | 12.9 | 10.1 | 3.8 | **26.7** |
| test, signbridge | 4,264 | 12.9 | 10.1 | 3.8 | **26.7** |

* Hypothesis normalization matches the official normalized CTM on all 1,161 sentences.
* With unit weights (plain Levenshtein) the total WER is the same, but the S/D/I split
  differs (dev 13.2/9.1/2.4).
* These numbers score the **release's example hypothesis**, not a model of ours.
* Tests: `tests/test_wer.py` (11 hand-computed cases) and `tests/test_phoenix_metric.py`
  (normalization cases, plus the sclite comparison marked `phoenix`).

## 5. Next steps (when PHOENIX work resumes)

1. Write `PhoenixDataset._build_records()` from the format above, using `fps=25` and
   `ImageFolderReader`.
2. Do the CTC fixes and the training loop (roadmap I-2, I-3), then the baseline on a GPU.
3. For BLEU, use the release's `sign-translation/bleu.py` or sacreBLEU, and say which one.
