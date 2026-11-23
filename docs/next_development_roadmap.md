# SignBridge: Next Development Roadmap

**Date:** 2026-11-23.

**Product target:** webcam → continuous VSL recognition → Vietnamese text → optional speech.

**Inputs:** [ai_validation_status.md](ai_validation_status.md),
[cslr_data_requirements.md](cslr_data_requirements.md),
[slt_data_requirements.md](slt_data_requirements.md),
[tts_implementation.md](tts_implementation.md).

## 1. End-to-end flow and the state of each stage

```
Webcam ──► Video preprocessing ──► CSLR ──► Gloss sequence ──► Vietnamese translation ──► Text ──► (TTS)
 [no]          [tested, real]        [blocked]   [blocked]            [blocked]           [n/a]   [tested]
```

| Stage | Implemented | Tested | Blocked by |
|---|---|---|---|
| Webcam capture for the CSLT path | **no**. The letter and ASL demos have one (`signbridge/live_io.py`), but it is not a `FrameSource`. | — | engineering only (small) |
| Video preprocessing (sampling, resize, windows, streaming) | yes | yes, real video | — |
| CSLR (video → glosses) | interface + untrained modules | shapes only | training code; gloss-annotated data (public: PHOENIX/GSL; Vietnamese: **none**) |
| Gloss sequence | data type exists (`GlossVocabulary`, `GlossOutput.decode`) | contract only | CSLR |
| Vietnamese translation | interface + untrained module | shapes only | **Vietnamese training data (none public)**; a gloss-input encoder; metrics |
| Text output | trivial | — | translation |
| TTS | **yes** (Piper, offline) | yes, 8 hand-written sentences; 28 tests | — (perceptual quality still to be rated) |

**Integration rule.** Untrained stages must **not** be chained into a demo that looks like it
translates. Until a stage has a measured result on held-out real data, the application
wiring must either:

* stop at the last validated stage; or
* label the output "untrained / not meaningful" and never show it to end users.

Today the only honest demo paths are:

* webcam → static VSL letters → typed text (`signbridge/letters/`), once trained on recorded letters;
* typed Vietnamese text → TTS;
* webcam → 29 isolated **ASL** greeting signs → phrase table → English/Vietnamese speech
  ([asl_greetings_demo.md](asl_greetings_demo.md)). The same pipeline can be retrained on
  VSL isolated signs once clips exist.

**Update 2026-11-23.**
* I-1 is partly done: WER with the official PHOENIX normalization reproduces sclite
  exactly ([phoenix_audit.md](phoenix_audit.md)).
* I-6 is done for the isolated-sign path: a webcam live loop.
* PHOENIX-2014-T is downloaded (dev, test and 1/8 of train). Its adapter is not written yet.

## 2. Current status (short)

* **Validated with real model output:**
  * the 29-sign ASL greetings demo, offline on held-out signers ([asl_greetings_demo.md](asl_greetings_demo.md));
  * TTS producing real Vietnamese speech, with quality not yet rated.
* **Validated as infrastructure:** video ingestion, preprocessing, manifests,
  video↔annotation checks.
* **Interfaces only:** CSLR and SLT models.
* **Missing:**
  * training/evaluation code (loop, WER, BLEU/chrF, checkpoints);
  * CTC beam search and window stitching;
  * a webcam `FrameSource`;
  * any Vietnamese continuous data.

## 3. Missing data and annotations

| Need | Exists publicly? | Path |
|---|---|---|
| Continuous VSL video + VSL gloss sequences | **no** | must be collected and annotated ([cslr_data_requirements §6](cslr_data_requirements.md)) |
| Continuous VSL video + Vietnamese sentences | **no** | collect alongside the above (same recordings, extra ELAN tier) |
| VSL gloss ↔ Vietnamese text pairs | not usable: one 10k corpus with no license or provenance | ask its authors for permission and documentation; or create pairs from our own annotations |
| Isolated VSL signs (pretraining, lexicon reference) | **yes**: VieSign-10K (open, CC BY 4.0); VSL400 (restricted, apply) | download VieSign-10K; apply for VSL400 |
| Public continuous gloss data (other sign languages) | **yes**: PHOENIX-2014-T, PHOENIX-2014, GSL | for pipeline validation only |

## 4. Recommended public datasets

| Purpose | Dataset | Why |
|---|---|---|
| First CSLR **and** SLT pipeline validation | **RWTH-PHOENIX-Weather 2014-T** | continuous signing, gloss and translation, direct download (39 GB), the standard benchmark with published WER/BLEU to sanity-check against; CC BY-NC-SA 3.0 (research only) |
| Alternative or second CSLR benchmark with a permissive license | **GSL** | CC BY 4.0, open on Zenodo; only 331 distinct sentences |
| Vietnamese isolated-sign pretraining and lexicon | **VieSign-10K** | open, CC BY 4.0, 5.9 GB, signer-independent subset |
| Larger Vietnamese isolated set | VSL400 | CC BY 4.0, but a Data Usage Agreement is required |

Not recommended now:

* How2Sign for CSLR (no gloss);
* CSL-Daily (needs a staff-signed institutional agreement);
* BOBSL (excludes independent researchers);
* any resource without a license.

## 5. Model development dependencies

```
metrics (WER, BLEU/chrF) ──┐
CTC fixes + feasibility ───┼──► training loop + checkpoints ──► PHOENIX CSLR baseline ──► beam search + window stitching ──► live CSLR
pretrained backbone load ──┤                                         │
PhoenixDataset adapter ────┘                                         └──► G2T / joint SLT on PHOENIX-2014-T (German)
GPU access ──────────────────────────────────────────────────────────────►(needed for the full runs)

VSL continuous recordings + gloss/translation annotation ──► VSL CSLR ──► VSL G2T ──► Vietnamese text ──► TTS (ready)
```

* **Compute.** This machine is CPU-only. Data preparation, unit tests, overfit checks and
  G2T (a small text model) are CPU-feasible. End-to-end RGB CSLR training needs a GPU
  (e.g. a cloud or university GPU). Pose-based encoders on MediaPipe keypoints are much
  cheaper and could run a small baseline on CPU.
* **Licenses:**
  * PHOENIX: CC BY-NC-SA; How2Sign: CC BY-NC. Research only.
  * Piper: GPL-3.0. Isolate it in its own process for product use.
  * VieSign-10K and GSL: CC BY 4.0.

## 6. Independent tasks that can be done now (no new data needed)

| # | Task | Size | Done when |
|---|---|---|---|
| I-1 | `signbridge/metrics/wer.py` (sub/del/ins) + chrF/BLEU via sacreBLEU, with hand-computed unit tests | S | tests pass on known cases |
| I-2 | CTC fixes: mask `<pad>`/`<unk>` in decoding; feasibility check with logging; native-fps CSLR config | S | unit tests for infeasible samples |
| I-3 | Training loop skeleton (`scripts/train_cslr.py`): config, seeding, AdamW, checkpoint/resume, metric logging | M | overfits a tiny batch of **real** isolated clips (e.g. VieSign-10K or ASL Citizen, as 1-gloss sequences, clearly labelled as a wiring test, not CSLR) |
| I-4 | Pretrained-weights option for `TorchvisionFrameEncoder` | S | loads ImageNet weights; shape test |
| I-5 | CTC prefix beam search decoder | M | matches greedy decoding at beam=1; unit tests |
| I-6 | Webcam `FrameSource` + `stream_windows` live loop (no model output shown as translation) | S–M | live windows logged with latency |
| I-7 | TTS: listen to and rate the 8 WAVs; optional ASR round-trip CER; text normalization for dates/abbreviations | S | ratings recorded in `tts_implementation.md` |
| I-8 | Gloss-input encoder for `TranslationModel` (gloss ids → `SequenceFeatures`) | S | shape tests; ready for G2T |
| I-9 | Annotation package: gloss guide, ELAN template, consent form draft, `continuous_csv` exporter | M | an ELAN file converts to a manifest that passes `validate_sample` |
| I-10 | Download VieSign-10K (5.9 GB, open) + adapter + audit, like How2Sign | M | audit doc with real counts |

## 7. Tasks blocked by missing data or resources

| Task | Blocked by |
|---|---|
| PHOENIX CSLR baseline with real WER | 39 GB download (allowed; just not done) + **GPU** |
| SLT pipeline validation (German) | the same, plus I-1 and I-8 |
| Any Vietnamese CSLR | continuous VSL recordings + VSL gloss annotation (**does not exist publicly**) |
| Vietnamese G2T | legitimate VSL gloss ↔ Vietnamese pairs (**none usable**) |
| Vietnamese Sign2Text | VSL video + Vietnamese sentences (**none**) |
| End-to-end webcam → Vietnamese speech demo | all of the above |
| VSL400 pretraining | Data Usage Agreement approval |

## 8. Prioritized roadmap

**Phase 0: harden what exists (1–2 weeks, CPU).**
* Do I-1, I-2, I-4, I-5 and I-8, then I-3 with the wiring test.
* Do I-7: rate TTS quality.
* Exit criterion: the training loop overfits real data, the metrics are unit-tested, and
  TTS has listener ratings.

**Phase 1: public-benchmark validation (2–4 weeks, needs a GPU).**
* Download PHOENIX-2014-T and audit it like How2Sign.
* Write `PhoenixDataset`.
* Train the CNN+TCN+BiLSTM+CTC baseline: sanity gates first, then the full run with
  3 seeds.
* Exit criterion: a dev/test WER from the official evaluation script, in the published
  baseline range or with any gap explained.
* Then train G2T (gloss → German) and joint SLT, and report BLEU/chrF on ground-truth and on
  predicted glosses.
* These results validate the **architecture**, not Vietnamese.

**Phase 2: Vietnamese data (start in parallel with Phase 0; longest lead time).**
* Download VieSign-10K. Apply for VSL400.
* Do I-9: build the annotation package (gloss guide, ELAN template, consent form).
* Recruit Deaf signers and annotators, and record a **pilot** of ~1,000 sentences with
  several signers. Annotate VSL glosses and Vietnamese translations, double-annotating
  ≥ 10 %.
* Ask the authors of the 10k Vietnamese–VSL text corpus for license and provenance.
* Exit criterion: a frozen, signer-independent VSL pilot manifest that passes
  `validate_sample`, with inter-annotator agreement reported.

**Phase 3: Vietnamese models.**
* Pretrain or initialize the visual encoder on VieSign-10K isolated signs.
* Train VSL CSLR on the pilot and report signer-independent WER.
* Train VSL G2T on the human gloss/translation pairs, and report BLEU and chrF on human
  glosses and on CSLR output.
* Human evaluation by Deaf VSL users.

**Phase 4: application.**
* Live webcam CSLR with window stitching and latency targets.
* Translation, then text, then Piper TTS in the existing FastAPI/Teams backend.
* Each stage is enabled in the UI only after it passes its Phase 1–3 exit criterion.
