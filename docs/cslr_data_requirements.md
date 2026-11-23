# CSLR: Data Requirements and Model Readiness

**Date:** 2026-11-23. **Scope:** which public data can supervise continuous sign language
recognition (video → gloss sequence), and whether the SignBridge CTC architecture is ready
to train on it.

**No CSLR model has been trained, and no WER of ours exists.** WER numbers below are
**published literature results**, quoted only to define what a sane baseline looks like.

How2Sign is **not** a candidate: its public release has no gloss annotations (see
[dataset_audit.md §1](dataset_audit.md)).

## 1. How the survey was done

* **Sources:** primary sources only (dataset pages, papers, license files, Zenodo records),
  read on 2026-11-23. No dataset was downloaded, no form was submitted, no account was
  created.
* **Spot checks:** these facts were re-checked directly:
  * Zenodo API: GSL record 3941811 (CC BY 4.0, open, 134.8 GB); VSL400 record 17943574
    (CC BY 4.0 but **restricted**); VieSign-10K record 23610105 (CC BY 4.0, open, 5.9 GB).
  * PHOENIX-2014-T page: 39 GB tarball, CC BY-NC-SA 3.0 badge, 210×260 frames.
* **"n/s"** below means the primary source does not state it.

## 2. Candidate datasets with continuous signing and gloss sequences

| Dataset | Language | Size | Gloss | Translation | Video | Access | License | Suitability |
|---|---|---|---|---|---|---|---|---|
| **RWTH-PHOENIX-Weather 2014-T** | DGS (German SL) | 8,257 sentence videos (7,096 / 519 / 642), 9 signers, 1,066 glosses, 2,887 German words | **yes** | **German** | PNG frames 210×260, 25 fps | direct download, 39 GB, no registration | CC BY-NC-SA 3.0 (non-commercial) | **Best first target.** The standard benchmark for both CSLR and SLT, so one dataset validates CTC and translation. |
| RWTH-PHOENIX-Weather 2014 | DGS | 6,841 sentences (5,672 / 540 / 629), 9 signers; vocab 1,081 or 1,295 depending on the source | yes | no | PNG frames 210×260, 25 fps | direct download, 53 GB | CC BY-NC-SA 3.0 | Very good for CSLR only. The most-used CSLR benchmark. |
| **GSL** (Greek SL) | GSL | 10,290 sentence instances of **331 unique sentences**, 310 glosses, 7 signers, 9.59 h | yes | Greek | RealSense 848×480, 30 fps (RGB + depth) | Zenodo, open, 134.8 GB including depth | **CC BY 4.0** (commercial allowed with attribution) | Good. The most permissive license. Few distinct sentences, so signer-independent WER is optimistic (6.1 % published). |
| CSL-Daily | Chinese SL | 20,654 videos (18,401 / 1,077 / 1,176), 10 signers, 2,000 glosses, ~23 h | yes | Chinese | 1920×1080, 30 fps | agreement signed by a **full-time staff member** of a university or institute, sent by email | research only | Good benchmark; access takes time and may exclude this project |
| USTC CSL (continuous) | Chinese SL | 100 sentences × 50 signers × 5 repetitions | sentence labels (gloss format n/s) | Chinese | 1280×720, 30 fps, plus depth/skeleton | same agreement as CSL-Daily | research only | Weak: only 100 sentences |
| Isharah | Saudi SL | ~30,000 clips, 18 signers | yes | Arabic (per project page) | n/s | registration + usage agreement | dataset license n/s | Possible; terms unverified |
| BOBSL (continuous-gloss subset) | BSL | manual continuous gloss for ~6 h of **test** video only | test only | English subtitles | MP4 (331 GB) | BBC terms of use, password | excludes commercial users and independent researchers | Poor for a first pipeline |
| Public DGS Corpus r4 | DGS | ~50 h conversations | German/English glosses (ELAN/iLex) | German/English | MP4 per transcript | direct per-file download | custom: "linguistic research" | Poor: heavy preprocessing; ML use not addressed |
| ASLLRP DAI 2 / NCSLGR | ASL | thousands of utterances | yes (SignStream) | English | n/s | web interface | research/education; no redistribution | Medium: bulk download path unclear |
| LSFB-CONT | French-Belgian SL | >25 h | per-hand gloss, JSON | French (ELAN) | MP4 720×576, 50 fps | email the author | n/s | Medium-low |

Not recommended:

* **Unclear license or hard-to-reach download:** CE-CSL (Chinese; no license stated,
  Baidu-Pan-only download).
* **Linguistic corpora:** Corpus NGT, SSL Corpus and Auslan Corpus are built for linguistics,
  and access is limited.

Detailed sources and URLs for each row are in §7.

### For contrast: continuous datasets **without** gloss (translation only)

| Dataset | Language | Note |
|---|---|---|
| How2Sign | ASL | >80 h, English; no public gloss (audited) |
| OpenASL | ASL | English, CC BY-NC-ND 4.0 |
| YouTube-ASL | ASL | 984 h of English captions; video IDs only |
| LSA-T | Argentine SL | ~22 h, Spanish; license n/s |
| ISL-CSLTR | Indian SL | 700 videos / 100 sentences, English, CC BY 4.0 |

These can train gloss-free Sign2Text, never CTC/CSLR.

## 3. Vietnamese Sign Language: what exists

**No public dataset of continuous VSL with gloss sequences could be found or verified.**
A 2026 peer-reviewed review of Vietnamese SLR reaches the same conclusion: existing VSL
corpora are almost entirely isolated signs (Hoang et al., *JST: Engineering and Technology
for Sustainable Development* 36(1), 2026).

| Resource | Type | Size | Access / license | Use for SignBridge |
|---|---|---|---|---|
| **VieSign-10K** (Zenodo 23610105) | isolated | 10,515 videos, 3,659 classes; signer-independent subset 447 classes × 50 signers | **open, CC BY 4.0**, 5.9 GB | best open VSL resource: isolated-sign pretraining, sign vocabulary reference |
| VSL400 (Zenodo 17943574) | isolated, 3 views | 74,259 clips, 400 glosses, 28 signers, ~54 h | CC BY 4.0 but **restricted** (Data Usage Agreement) | apply for access; strong isolated pretraining set |
| Multi-VSL (WACV 2025) | isolated, multi-view | 84,764 videos, 1,000 glosses, 30 signers | paper says public; repository URL not found. A pose-only HF mirror has license "unknown" | do not use until the official source and license are confirmed |
| SignConnect (HF) | isolated (QIPEDC dictionary videos) | 4,362 videos | card says MIT; rights to the dictionary videos unverified | reference only |
| Corpus-Vie-VSL-10k (GitHub) | **text-only** Vietnamese ↔ VSL-gloss pairs, 10k sentences | — | **no license file, provenance unstated** | do not reuse without the author's written permission |
| Own letter recordings (`scripts/record_letters.py`) | static fingerspelling, landmarks only | as recorded | own data | letter classifier only |

## 4. Readiness of the existing RGB + temporal encoder + CTC architecture

Reviewed: `signbridge/models/{interfaces,reference}.py`, `signbridge/text/vocab.py`,
`signbridge/data/collate.py`, `signbridge/video/`.

| Question | Finding |
|---|---|
| **Ready for supervised training?** | **No, but the core pieces exist.** There is no training loop, optimizer/scheduler, checkpointing, metric, or experiment config. The model and data contracts are in place. |
| **CTC targets correct?** | Yes for the format: `collate` pads gloss ids to (B, S) and returns `gloss_lengths`. `CTCGlossDecoder.loss` passes `log_probs (T,B,V)`, the downsampled input lengths and target lengths to `nn.CTCLoss(blank=0)`. That matches the PyTorch contract and is tested. |
| **Vocabulary correct?** | Mostly. `<blank>` = 0 is reserved as required. **Issues to fix:** (a) `<pad>` and `<unk>` are output classes the model can emit; mask them in decoding, or exclude them from the output layer. (b) Unknown dev/test glosses map to `<unk>` in targets. WER must be computed against the **original** reference strings, not `<unk>`-mapped ids. (c) Dataset-specific gloss cleanup (e.g. PHOENIX `__ON__`/`__OFF__`, `loc-` prefixes) must follow the dataset's **official evaluation script**, or WER is not comparable with the literature. |
| **CTC feasibility** | **Gap.** `nn.CTCLoss(zero_infinity=True)` silently zeroes samples where the encoder output is too short (T' < target length + repeats). With the default 12 fps sampling and 4× temporal downsampling, that is 3 steps per second, which can be too few for fast signing. **Add a feasibility check** that counts and logs infeasible samples, and **use the native frame rate** (25 fps for PHOENIX) for CSLR. |
| **Backbone** | `TorchvisionFrameEncoder` defaults to `weights=None` (random). CSLR baselines use ImageNet-pretrained ResNet-18/34. Weight loading must be added. `FrameCNNEncoder` is a toy. |
| **Compute** | This machine is CPU-only (torch `+cpu`). End-to-end RGB CSLR on PHOENIX needs a GPU (typically 1 GPU for 40–80 epochs in published baselines). CPU is enough for data preparation, unit tests and a tiny overfit check. |
| **Reproducibility** | The data side is reproducible: manifest SHA-256 identical across builds, and seeded forwards. The training side does not exist yet. Needed: global seeding (`torch`, `numpy`, `random`, DataLoader workers), deterministic split manifests, a config snapshot plus git commit hash per run, and checkpoint files with metrics. |
| **Continuous video inference** | Partly. Offline, a whole clip is processed with `VideoPreprocessor.process` and decoded in one pass. Live or long-stream inference has `stream_windows()` for bounded-memory windows, but **no CTC stitching across overlapping windows** and no webcam `FrameSource` for this path. A BiLSTM/Transformer temporal encoder is non-causal, so streaming needs chunked context or a causal variant. |
| **Data input** | PHOENIX ships PNG frame folders. `ImageFolderReader(folder, pattern, fps=25)` already reads that layout. A `PhoenixDataset._build_records()` adapter is needed. It maps the corpus file rows to `ManifestRecord`: gloss string → `gloss_sequence`, signer → `signer_id`, German sentence → `target_text` (2014-T). Verify the exact file layout after download, because sources disagree (official page: XML; community code: pipe-separated `corpus.csv`). |

## 5. First valid CSLR experiment: implementation sequence

Goal: a real, reproducible WER on a public benchmark **using official labels**. It
validates the architecture, not Vietnamese. Recommended dataset: **PHOENIX-2014-T** (it
also has German translations for the SLT stage). Use **GSL** instead if the CC BY 4.0
license matters.

1. **Download and verify.** Fetch `phoenix-2014-T.v3.tar.gz` (39 GB) outside the repository
   (`../datasets/PHOENIX-2014-T/`). Record SHA-256 in a provenance file. Inspect the annotation
   files **before** writing the adapter (layout, delimiter, gloss conventions).
2. **Adapter.** Write `signbridge/data/phoenix.py` (`PhoenixDataset`): emit `ManifestRecord`s
   with `gloss_sequence`, `signer_id`, `target_text`, `fps=25` from the documentation, and
   status `gloss_and_translation`. Use the official train/dev/test splits unchanged. Add an
   audit script and real-data tests, like the How2Sign ones.
3. **Metrics first.** Implement `signbridge/metrics/wer.py` (Levenshtein with
   substitution/deletion/insertion counts). Unit-test it on hand-computed cases. Wrap the
   official PHOENIX evaluation (gloss cleanup + scoring) so reported numbers are comparable.
4. **Vocabulary and CTC fixes.** Mask `<pad>`/`<unk>` in decoding. Add the CTC feasibility
   check. Use a native-fps (25 fps) preprocessing config with augmentation: random crop,
   temporal rescale ±20 %.
5. **Model.** ImageNet-pretrained ResNet-18 frame encoder → `TemporalConvEncoder`
   (downsample 4) → BiLSTM/Transformer → `CTCGlossDecoder`. This mirrors the widely published
   CNN+TCN+BiLSTM+CTC baseline.
6. **Training script.** Write `scripts/train_cslr.py`:
   * AdamW, cosine or step LR;
   * gradient clipping;
   * mixed precision on GPU;
   * checkpoint best-dev-WER;
   * log config, git hash, seed and data manifest hash.
7. **Sanity gates before the full run:**
   * (a) Overfit 20 training samples to near-zero loss. This proves labels and the loss are
     wired correctly.
   * (b) Confirm dev WER beats trivial baselines: an empty hypothesis gives 100 %, and a
     most-frequent-gloss output is also scored.
   * (c) Confirm the number of infeasible CTC samples is 0.
8. **Full run** on GPU. Report dev/test WER with ≥ 3 seeds (mean ± std), plus
   substitution/deletion/insertion rates. Published baselines put a plain CNN+BiLSTM+CTC
   around 22–26 % WER on PHOENIX and state of the art around 18–21 % (CorrNet, CVPR 2023;
   TwoStream-SLR, NeurIPS 2022). Treat a result far outside that range as a bug signal until
   explained.
9. **Continuous inference.** Add a beam-search CTC decoder and windowed inference with
   overlap stitching. Measure latency and WER degradation against whole-clip decoding on the
   same test set.

Only after this works does Vietnamese data become the bottleneck (§6).

## 6. Creating a continuous VSL gloss dataset (no public one exists)

### 6.1 Ethics and consent

* Obtain written consent covering **video recording, research use, and redistribution or
  publication** (separately).
* Faces are essential for sign language, so the data cannot be anonymized. Store it
  encrypted and access-controlled.
* Pay Deaf signers and annotators fairly. Involve the Deaf community in the design, for
  example Hội Người Điếc and local Deaf clubs.

### 6.2 Recording

* **Signers:** at least 10, so held-out signers are possible. Balance region (Hà Nội /
  Hải Phòng / TP.HCM), gender and age. Record region and signer ID per clip.
* **Content:** sentences from target domains (daily conversation, healthcare, public
  services) written or adapted **by Deaf signers**. Also record natural signing, not only
  scripted signing.
* **Capture:** frontal camera at 1080p, ≥ 30 fps (ideally 50–60 fps for fast hands); plain
  background; full upper body; optional second view. Use a fixed file naming scheme.

### 6.3 Annotation

* **Tool:** ELAN, the standard for sign language.
* **Tiers:**
  * sentence boundaries;
  * **VSL gloss** (ID-gloss);
  * Vietnamese translation;
  * optional non-manual markers (e.g. negation, question).
* **Gloss conventions:** write a **gloss guide** with ID-glosses, upper-case convention and
  markers for fingerspelling (`FS:`), classifiers (`CL:`), pointing (`IX`), name signs and
  numbers. Base the ID-glosses on a reference lexicon, e.g. the QIPEDC dictionary or
  VieSign-10K classes, without copying their videos.
* **Annotators:** **Deaf VSL signers or certified interpreters**. Glosses are annotated
  from the video, never derived from the Vietnamese sentence.
* **Quality control:**
  * double-annotate ≥ 10 % of sentences;
  * report inter-annotator agreement: gloss-sequence WER between annotators, and boundary
    IoU;
  * adjudicate disagreements and update the gloss guide.
* **Export:** ELAN → the proposed `continuous_csv` layout (columns `sample_id, video_path,
  split, signer_id, gloss, vietnamese_text, start_time, end_time`), then run
  `build_manifests.py` and `validate_sample` on every record.

### 6.4 Size targets (rough)

| Target | Data |
|---|---|
| A **pilot** that can detect pipeline bugs and give a first signer-independent number | ~1,000 sentences × several signers |
| PHOENIX-scale | 5,000–8,000 sentences, ~1,000 glosses |

### 6.5 Splits

Signer-independent test and dev sets, frozen and hashed before any training.

## 7. Sources

| Topic | URL |
|---|---|
| PHOENIX-2014-T | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ |
| PHOENIX-2014-T counts | Camgöz et al. 2020, https://arxiv.org/abs/2003.13830 |
| PHOENIX-2014 | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/ (Koller et al., CVIU 2015) |
| PHOENIX-2014 community preprocessing (`corpus.csv` format) | https://github.com/ycmin95/VAC_CSLR |
| GSL | https://vcl.iti.gr/dataset/gsl/ · https://zenodo.org/record/3941811 · arXiv:2007.12530 |
| CSL-Daily | https://ustc-slr.github.io/datasets/2021_csl_daily/ · arXiv:2105.12397 |
| USTC CSL | https://ustc-slr.github.io/datasets/2015_csl/ |
| Isharah | https://snalyami.github.io/Isharah_CSLR/ · arXiv:2506.03615 |
| BOBSL | https://www.robots.ox.ac.uk/~vgg/data/bobsl/ · arXiv:2405.10266 |
| Public DGS Corpus | https://www.sign-lang.uni-hamburg.de/meinedgs_r4/ling/license_en.html |
| ASLLRP DAI | https://dai.cs.rutgers.edu/dai/s/dai · https://www.bu.edu/asllrp/dai-terms.html |
| LSFB-CONT | https://lsfb-team.github.io/lsfb-dataset/lsfb_cont |
| CE-CSL | arXiv:2409.11960 · https://github.com/woshisad159/TFNet |
| VieSign-10K | https://zenodo.org/records/23610105 |
| VSL400 | https://zenodo.org/records/17943574 |
| Multi-VSL | https://mlanthology.org/wacv/2025/dinh2025wacv-sign/ |
| VSL review | https://jst.vn/index.php/etsd/article/download/1293/1315 |
| Corpus-Vie-VSL-10k | https://github.com/datheobc123/Vietnamese-Text-To-Sign-Lanuage- |
| Published WER: CorrNet | arXiv:2303.03202 |
| Published WER: TwoStream-SLR | arXiv:2211.01367 |

### Could not verify

* **PHOENIX:** the exact 2014-T annotation layout, and the gloss vocabulary size of
  PHOENIX-2014 (1,081 vs 1,295).
* **GSL:** the annotation file format, and whether RGB is stored as frames or MP4.
* **CSL-Daily:** the full agreement terms.
* **Licenses not stated:** CE-CSL, LSA-T, LSFB-CONT, YouTube-ASL, Isharah.
* **Multi-VSL:** the official repository URL.
* **Corpus-Vie-VSL-10k:** column layout and provenance.
