# SLT (Sign Language → Vietnamese Text): Data Requirements and Readiness

**Date:** 2026-11-23. No translation model has been trained, and no translation-quality
number exists. Output from the untrained reference model is never used as evidence.

## 1. Three different problems

| Approach | Input → output | Training pairs required | Strength | Weakness |
|---|---|---|---|---|
| **Gloss-to-text (G2T)** | gloss sequence → Vietnamese sentence | (VSL gloss sequence, Vietnamese sentence) | small text-only model; trainable and testable **without video**; errors are easy to inspect | needs gloss annotation; glosses drop information the video contains (facial grammar, classifiers, space) |
| **Video-to-text (Sign2Text, gloss-free)** | video → Vietnamese sentence | (video, Vietnamese sentence) | no gloss annotation needed | needs much more data; hardest to debug; How2Sign-style data shows this is difficult even at 80 h |
| **Recognition then translation (Sign2Gloss2Text)** | video → CSLR → glosses → G2T → sentence | (video, gloss) **and** (gloss, sentence); ideally triples (video, gloss, sentence) | modular: each stage is measured on its own; G2T can use extra text-only gloss data | recognition errors propagate; two annotation layers |
| Joint (Sign2(Gloss+Text)) | shared encoder, CTC + translation heads | triples (video, gloss, sentence) | usually the strongest configuration in the literature | needs the same triples as Sign2Gloss2Text |

SignBridge's interfaces already cover all three (`SignTranslationPipeline` gloss-free vs.
gloss modes, [architecture.md §4](architecture.md)). The **data** decides which one can
actually be trained.

## 2. Current state of the translation component

| Item | State |
|---|---|
| `TranslationModel` contract: `forward(memory, target_in)`, `generate(memory)` | implemented; memory is a `SequenceFeatures` |
| `TransformerTranslationModel` (reference) | untrained; shape and generation-loop checks only |
| `TextVocabulary` | Unicode word tokenizer that keeps Vietnamese diacritics; `<pad>/<unk>/<bos>/<eos>`. Word-level only: no subword model. |
| Gloss → text input path | **missing**. `TranslationModel` takes visual features as memory. A G2T model needs a gloss embedding encoder that emits `SequenceFeatures`. That is a small addition: embed gloss ids → `SequenceFeatures`, then the same decoder. |
| Metrics | **missing**: no BLEU / chrF / ROUGE implementation or evaluation script |
| Vietnamese training data | **none** in the repository |

## 3. Required data for Vietnamese output

The minimum for each approach:

* **G2T:** (VSL gloss sequence, Vietnamese sentence) pairs, written by annotators who know
  VSL. Glosses must come from the signing. Never derive glosses mechanically from the
  Vietnamese sentence: that produces Vietnamese word order, not VSL, and the model would
  learn an identity-like mapping.
* **Sign2Text:** VSL videos with sentence-level Vietnamese translations, sentence-aligned in
  time.
* **Sign2Gloss2Text / joint:** both of the above on the same videos (triples).

For all three:

* **Signer IDs.** Record signer IDs so test signers are unseen at training time.
* **Regional variant.** Record the variant (Hà Nội / Hải Phòng / TP.HCM), because VSL has
  strong regional variation.
* **Split hygiene.** Freeze train/dev/test splits **before** training. No sentence and no
  signer appears in more than one split, unless an experiment explicitly tests seen signers.

**Public availability (searched 2026-11-23; details in
[cslr_data_requirements.md §3](cslr_data_requirements.md)):**

* **VSL video with Vietnamese sentence translations:** none found. Every public VSL video
  dataset found is **isolated signs**, e.g. VieSign-10K (open, CC BY 4.0) and VSL400
  (restricted access).
* **VSL gloss ↔ Vietnamese text pairs:**
  * A 10k-pair corpus is described in a 2023 paper (Nguyen, Phung, Vu, *Journal of Computer
    Science and Cybernetics* 39(2)) with no public link.
  * A GitHub repository ("Corpus-Vie-VSL-10k") offers a 10,000-sentence CSV/XLSX but has
    **no license and no stated provenance**. Its VSL side was not checked for being genuine
    VSL glosses rather than re-ordered Vietnamese.
  * It is **not usable** without the author's written permission and a check of how the
    glosses were made.
* **Public gloss/text corpora in other languages:** usable for pipeline validation only.
  * PHOENIX-2014-T: DGS → German, CC BY-NC-SA 3.0, direct download.
  * GSL: Greek SL → Greek, CC BY 4.0.
  * CSL-Daily: CSL → Chinese, institutional agreement.

## 4. Can G2T be developed independently of CSLR?

**Yes, if and only if there are legitimate (VSL gloss, Vietnamese) pairs.** G2T is a pure text
task. It can be:

* trained on gloss annotations produced by humans;
* evaluated against human reference translations;
* improved before any CSLR model exists.

At inference time it then receives CSLR output instead of human glosses.

Not legitimate as G2T training data:

* ASL or other-language glosses mapped to Vietnamese;
* glosses generated from Vietnamese text by rules or an LLM;
* "glosses" that are only the Vietnamese sentence tokenized.

Synthetic gloss augmentation (e.g. back-translation) is a research technique that is only
valid **in addition to** a real, human-annotated dev/test set, and must be reported as such.

A useful intermediate is to develop the G2T pipeline on a public **non-Vietnamese** gloss/text
corpus (PHOENIX-2014-T DGS→German, CSL-Daily CSL→Chinese). This validates training, decoding
and metrics with real data. Those results say **nothing** about Vietnamese quality and are
reported as pipeline validation only.

## 5. Metrics and validation procedure

| Metric | Use | Notes |
|---|---|---|
| **BLEU-4** (sacreBLEU) | primary, for comparability with the SLT literature | Report the sacreBLEU signature. Vietnamese needs fixed tokenization. Corpus-level, never sentence-level averages. |
| **chrF / chrF++** | primary for Vietnamese | character-level, robust to tokenization and diacritic variation; recommended alongside BLEU |
| **ROUGE-L** | secondary | reported in PHOENIX-2014-T papers |
| BLEURT / COMET (multilingual) | optional | only with a model that supports Vietnamese; report the model version |
| **Human evaluation** | required before any product claim | adequacy and fluency (1–5) by native Vietnamese speakers, ideally Deaf VSL users; blind to the system |

Procedure:

1. **Freeze the test set first.** Fix and version the test split (hash the manifest) before
   training. Use the dev set for model selection, and evaluate on test **once** per reported
   configuration.
2. **Report baselines:**
   * for G2T, a copy/identity baseline (gloss tokens joined as text);
   * the reference translations' own BLEU against a second reference, if two exist;
   * a retrieval baseline (nearest training sentence).
3. **Run several seeds:** at least 3 seeds; report mean ± std.
4. **Report the full context:** data size, signer split and decoding settings (beam size,
   length penalty).
5. **Analyze errors** on a sample of at least 100 test sentences: omissions, hallucinations,
   word order, negation, numbers and names.
6. **End to end:** for Sign2Gloss2Text, report CSLR WER, then G2T BLEU on **ground-truth**
   glosses, then BLEU on **predicted** glosses. The gap measures error propagation.

## 6. Readiness verdict

| Question | Answer |
|---|---|
| Can a Vietnamese translation model be trained today? | **No.** There is no Vietnamese sign/gloss/text training data in the project. |
| Can the translation *pipeline* (training loop, decoding, metrics) be built and validated today? | **Yes**, on a public gloss/text corpus in another language, labelled as pipeline validation |
| What unblocks Vietnamese SLT? | VSL recordings with Vietnamese sentence translations; for the modular route, VSL gloss annotation too |
