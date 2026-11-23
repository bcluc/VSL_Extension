"""Integration tests on the downloaded How2Sign val subset. Skipped when the subset is
absent (see scripts/fetch_how2sign_subset.py). These verify facts documented in
docs/dataset_audit.md so the documentation cannot silently drift from the data."""
import pytest

from signbridge.data import How2SignDataset
from signbridge.data.validation import locate_clip_in_video, validate_sample
from signbridge.video import PreprocessConfig, VideoPreprocessor

pytestmark = pytest.mark.realdata


def test_val_annotation_counts_and_no_gloss(h2s_root):
    recs = How2SignDataset(h2s_root, "val", "clips", probe_videos=False).load_manifest()
    assert len(recs) == 1741
    assert len({r.meta["video_name"] for r in recs}) == 132
    assert all(r.gloss_sequence is None and r.annotation_status == "translation_only" for r in recs)
    assert all(r.target_text for r in recs)


def test_csv_quoted_sentences_are_decoded(h2s_root):
    recs = How2SignDataset(h2s_root, "val", "clips", probe_videos=False).load_manifest()
    quoted = [r.target_text for r in recs if '""' in r.target_text or r.target_text.startswith('"')]
    assert quoted == []


def test_downloaded_clips_decode_and_have_measured_fps(h2s_root):
    ds = How2SignDataset(h2s_root, "val", "clips")
    avail = ds.sample_ids(available_only=True)
    assert len(avail) >= 1
    for sid in avail[:5]:
        chk = validate_sample(ds, sid)
        assert chk.ok, chk.errors
        assert ds.get_sample(sid).record.fps in (24.0, 30.0, 60.0)


def test_clips_match_original_not_realigned_timestamps(h2s_root):
    full = How2SignDataset(h2s_root, "val", "original_full")
    rows = [r for r in full.load_manifest() if r.video_available]
    if not rows:
        pytest.skip("no full video downloaded")
    clips = How2SignDataset(h2s_root, "val", "clips")
    r = next(r for r in rows if clips.clip_path(r.meta["sentence_name"]).exists())
    clip = clips.clip_path(r.meta["sentence_name"])
    orig = locate_clip_in_video(clip, r.video_path, r.meta["original_start"], search_s=2.0)
    assert abs(orig["offset_frames"]) <= 1
    real = locate_clip_in_video(clip, r.video_path, r.meta["realigned_start"], search_s=2.0)
    assert real["mad_at_expected"] > orig["mad_best"] * 1.3


def test_realigned_segments_inside_full_video(h2s_root):
    ds = How2SignDataset(h2s_root, "val", "realigned_full")
    for sid in ds.sample_ids(available_only=True):
        assert validate_sample(ds, sid).ok


def test_container_seek_is_frame_exact_on_real_h264(h2s_root):
    import numpy as np
    from signbridge.video import VideoReader
    ds = How2SignDataset(h2s_root, "val", "realigned_full")
    rows = [r for r in ds.load_manifest() if r.video_available]
    if not rows:
        pytest.skip("no full video downloaded")
    r = max(rows, key=lambda r: r.start_time)
    a = [f for _, f in VideoReader(r.video_path, r.start_time, r.end_time, seek="seek").read_frames()]
    b = [f for _, f in VideoReader(r.video_path, r.start_time, r.end_time, seek="grab").read_frames()]
    assert len(a) == len(b) > 0 and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_preprocess_real_clip(h2s_root):
    ds = How2SignDataset(h2s_root, "val", "clips")
    sid = ds.sample_ids(available_only=True)[0]
    clip = VideoPreprocessor(PreprocessConfig(target_fps=12, image_size=(112, 112))).process(ds.load_video(sid))
    assert clip.frames.shape[1:] == (3, 112, 112) and clip.valid_mask.all()
