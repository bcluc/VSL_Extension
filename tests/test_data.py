import json

import numpy as np
import pytest

from signbridge.data import (AnnotationMissingError, AnnotationStatus, How2SignDataset, ManifestDataset,
                             ManifestRecord, SampleNotFoundError, VSLDataset, read_manifest, write_manifest,
                             write_manifest_csv)
from signbridge.data.collate import collate, pad_sequences
from signbridge.data.validation import locate_clip_in_video, validate_sample
from signbridge.text import GlossVocabulary, TextVocabulary, tokenize_text
from signbridge.video import VideoUnavailableError


# ---------------------------------------------------------------- schema
def _rec(**kw):
    base = dict(sample_id="s1", dataset_name="x", language="asl", video_path="v.mp4", split="val",
                annotation_status="translation_only", target_text="hi")
    base.update(kw)
    return ManifestRecord(**base)


def test_schema_defaults_are_null():
    r = _rec()
    d = r.to_dict()
    for k in ("signer_id", "gloss_sequence", "source_text", "start_time", "end_time", "fps"):
        assert d[k] is None


def test_schema_validation():
    with pytest.raises(ValueError):
        _rec(annotation_status="made_up")
    with pytest.raises(ValueError):
        _rec(start_time=2.0, end_time=1.0)
    with pytest.raises(ValueError):
        _rec(language="xx")
    with pytest.raises(ValueError):
        _rec(gloss_sequence=["A", ""])


def test_status_inference():
    assert AnnotationStatus.infer(None, "t") is AnnotationStatus.TRANSLATION_ONLY
    assert AnnotationStatus.infer(["A"], "t") is AnnotationStatus.GLOSS_AND_TRANSLATION
    assert AnnotationStatus.infer(["A"], None) is AnnotationStatus.GLOSS_ONLY
    assert AnnotationStatus.infer(None, None) is AnnotationStatus.UNANNOTATED
    assert AnnotationStatus.infer(["A"], None, isolated=True) is AnnotationStatus.ISOLATED_LABEL


def test_manifest_roundtrip_jsonl_and_csv(tmp_path):
    recs = [_rec(), _rec(sample_id="s2", gloss_sequence=["A", "B"], annotation_status="gloss_and_translation",
                         meta={"k": 1})]
    p = write_manifest(recs, tmp_path / "m.jsonl")
    assert read_manifest(p) == recs
    assert json.loads(p.read_text().splitlines()[0])["fps"] is None          # null, not fabricated
    csv_text = write_manifest_csv(recs, tmp_path / "m.csv").read_text(encoding="utf-8")
    assert "A B" in csv_text
    assert len(ManifestDataset(p)) == 2


# ---------------------------------------------------------------- How2Sign adapter (synthetic tree)
def test_how2sign_clips_view(synth_how2sign):
    ds = How2SignDataset(synth_how2sign, "val", "clips")
    recs = ds.load_manifest()
    assert [r.sample_id for r in recs] == ["vidA_0-1-rgb_front", "vidA_1-1-rgb_front", "vidA_2-1-rgb_front"]
    r0 = recs[0]
    assert r0.language == "asl" and r0.dataset_name == "how2sign"
    assert r0.gloss_sequence is None and r0.source_text is None and r0.signer_id is None
    assert r0.target_text == 'He said "ok" here.'           # unquoted field: quotes kept literally
    assert recs[1].target_text == 'An "i" here.'             # CSV-quoted field, doubled quotes decoded
    assert r0.annotation_status == "translation_only"
    assert r0.start_time is None and r0.end_time is None   # whole clip
    assert r0.fps == pytest.approx(20.0)                     # measured, video exists
    assert r0.meta["original_start"] == 0.0 and r0.meta["realigned_start"] == 0.1
    missing = recs[2]
    assert missing.meta["video_available"] is False and missing.fps is None
    assert ds.sample_ids(available_only=True) == ["vidA_0-1-rgb_front", "vidA_1-1-rgb_front"]


def test_how2sign_interface_methods(synth_how2sign):
    ds = How2SignDataset(synth_how2sign)
    s = ds.get_sample("vidA_0-1-rgb_front")
    assert s.annotation.require_translation().startswith("He said")
    with pytest.raises(AnnotationMissingError):
        s.annotation.require_gloss()
    with ds.load_video("vidA_0-1-rgb_front") as v:
        assert v.num_frames == 20
    md = ds.get_metadata("vidA_0-1-rgb_front")
    assert md["video"]["frame_count"] == 20 and "CC BY-NC" in md["license"]
    with pytest.raises(VideoUnavailableError):
        ds.load_video("vidA_2-1-rgb_front")
    with pytest.raises(SampleNotFoundError):
        ds.get_sample("nope")


def test_how2sign_full_video_views(synth_how2sign):
    re = How2SignDataset(synth_how2sign, view="realigned_full").load_manifest()
    assert re[0].sample_id == "vidA_0-1-rgb_front#realigned"
    assert (re[0].start_time, re[0].end_time) == (0.1, 0.9)
    assert re[0].video_path.endswith("vidA-1-rgb_front.mp4")
    ds = How2SignDataset(synth_how2sign, view="original_full")
    v = ds.load_video("vidA_1-1-rgb_front#original")
    assert (v.start_frame, v.end_frame) == (20, 40)


def test_how2sign_rejects_bad_schema(tmp_path, synth_how2sign):
    p = synth_how2sign / "sentence_level/val/text/en/raw_text/how2sign_val.csv"
    p.write_text("A\tB\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected columns"):
        How2SignDataset(synth_how2sign).load_manifest()


def test_how2sign_missing_annotation_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        How2SignDataset(tmp_path).load_manifest()


# ---------------------------------------------------------------- VSL adapter
def test_vsl_letter_frames(synth_vsl_letters):
    ds = VSLDataset(synth_vsl_letters)
    recs = ds.load_manifest()
    assert [r.gloss_sequence for r in recs] == [["A"], ["Đ"], ["Mũ"]]
    assert all(r.annotation_status == "isolated_label" and r.fps is None and r.target_text is None for r in recs)
    src = ds.load_video("letters/DD")
    assert src.num_frames == 6
    assert validate_sample(ds, "letters/A").ok


def test_vsl_continuous_csv(tmp_path, synth_video):
    root = tmp_path / "vsl_cont"
    root.mkdir()
    (root / "manifest.csv").write_text(
        "sample_id,video_path,split,signer_id,gloss,vietnamese_text,start_time,end_time\n"
        f"c1,{synth_video},train,S01,XIN CHÀO,Xin chào bạn,0.0,1.0\n"
        f"c2,{synth_video},train,,,Cảm ơn,,\n"
        f"c3,missing.mp4,test,,,,,\n", encoding="utf-8")
    ds = VSLDataset(root, split="all", layout="continuous_csv")
    r1, r2, r3 = ds.load_manifest()
    assert r1.gloss_sequence == ["XIN", "CHÀO"] and r1.annotation_status == "gloss_and_translation"
    assert r1.signer_id == "S01" and r1.fps == pytest.approx(20.0)
    assert r2.gloss_sequence is None and r2.annotation_status == "translation_only" and r2.start_time is None
    assert r3.annotation_status == "unannotated" and r3.meta["video_available"] is False
    assert [r.sample_id for r in VSLDataset(root, split="train", layout="continuous_csv").load_manifest()] == ["c1", "c2"]


def test_same_pipeline_code_for_both_datasets(synth_how2sign, synth_vsl_letters):
    """The preprocessing/batching code path is identical for ASL clips and VSL frames."""
    from signbridge.video import PreprocessConfig, VideoPreprocessor
    pre = VideoPreprocessor(PreprocessConfig(sampling="uniform", num_samples=6, image_size=(16, 16),
                                             window_size=4, window_stride=2))
    items = []
    for ds, sid in ((How2SignDataset(synth_how2sign), "vidA_0-1-rgb_front"),
                    (VSLDataset(synth_vsl_letters), "letters/A")):
        clip = pre.process(ds.load_video(sid))
        items.append({"record": ds.get_sample(sid).record, "inputs": clip.frames, "valid_mask": clip.valid_mask})
    b = collate(items)
    assert b.inputs.shape == (2, 6, 3, 16, 16)
    assert b.annotation_status == ["translation_only", "isolated_label"]


# ---------------------------------------------------------------- vocab + collate
def test_tokenize_keeps_vietnamese_diacritics():
    assert tokenize_text("Xin chào, bạn khỏe không?") == ["xin", "chào", ",", "bạn", "khỏe", "không", "?"]


def test_gloss_vocab_requires_real_glosses():
    with pytest.raises(ValueError, match="no gloss annotations"):
        GlossVocabulary.from_sequences([None, None])
    v = GlossVocabulary.from_sequences([["B", "A"], ["A"]])
    assert v.itos[:3] == ["<blank>", "<pad>", "<unk>"] and v.encode(["A", "Z"]) == [3, 2]


def test_pad_sequences():
    out, lengths, mask = pad_sequences([np.ones((3, 2)), np.ones((5, 2))])
    assert out.shape == (2, 5, 2) and lengths.tolist() == [3, 5]
    assert mask.sum(1).tolist() == [3, 5] and out[0, 3:].sum() == 0
    with pytest.raises(ValueError):
        pad_sequences([np.ones((3, 2)), np.ones((3, 4))])


def test_collate_with_missing_gloss_is_explicit():
    recs = [_rec(sample_id="a", target_text="hello world"),
            _rec(sample_id="b", gloss_sequence=["X"], annotation_status="gloss_and_translation", target_text="hi")]
    items = [{"record": r, "inputs": np.zeros((n, 4), np.float32)} for r, n in zip(recs, (3, 6))]
    gv, tv = GlossVocabulary(["X"]), TextVocabulary.from_texts([r.target_text for r in recs])
    b = collate(items, gloss_vocab=gv, text_vocab=tv)
    assert b.has_gloss.tolist() == [False, True] and b.gloss_lengths.tolist() == [0, 1]
    assert b.text_ids[0, 0] == tv.bos_id and b.text_lengths.tolist() == [4, 3]
    with pytest.raises(AnnotationMissingError):
        collate(items, require_gloss=True)
    t = b.to_torch()
    assert t["inputs"].shape == (2, 6, 4) and t["input_lengths"].tolist() == [3, 6]


# ---------------------------------------------------------------- validation
def test_validate_sample_reports_missing_video(synth_how2sign):
    ds = How2SignDataset(synth_how2sign)
    chk = validate_sample(ds, "vidA_2-1-rgb_front")
    assert not chk.ok and "not available" in chk.errors[0]
    ok = validate_sample(ds, "vidA_0-1-rgb_front", full_decode=True)
    assert ok.ok and ok.info["decoded_frames"] == 20 and ok.info["clip_minus_annotation_s"] == 0.0


def test_validate_segment_beyond_video(synth_how2sign):
    p = synth_how2sign / "sentence_level/val/text/en/raw_text/re_aligned/how2sign_realigned_val.csv"
    txt = p.read_text(encoding="utf-8").replace("3.4\t", "9.0\t")     # end beyond the 4 s full video
    p.write_text(txt, encoding="utf-8")
    ds = How2SignDataset(synth_how2sign, view="realigned_full")
    chk = validate_sample(ds, "vidA_2-1-rgb_front#realigned")
    assert not chk.ok and any("beyond video end" in e for e in chk.errors)


def test_locate_clip_in_video(synth_how2sign):
    clip = synth_how2sign / "sentence_level/val/rgb_front/raw_videos/vidA_0-1-rgb_front.mp4"
    full = synth_how2sign / "video_level/val/rgb_front/raw_videos/vidA-1-rgb_front.mp4"
    res = locate_clip_in_video(clip, full, expected_start=0.0, search_s=1.0, n_probe=4)
    assert res["offset_frames"] == 0 and res["mad_best"] < 1.0
