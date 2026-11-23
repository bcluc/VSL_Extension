import numpy as np
import pytest

from conftest import frame_level, write_video
from signbridge.video import (CorruptVideoError, ImageFolderReader, PreprocessConfig, VideoPreprocessor,
                              VideoReader, VideoUnavailableError, make_windows, probe_video, sample_indices)
from signbridge.video.sampling import interval_iou, time_to_frame_range
from signbridge.video.transforms import resize_frame


# ---------------------------------------------------------------- probing / reading
def test_probe_reports_fps_frames_duration(synth_video):
    info = probe_video(synth_video, full_decode=True)
    assert info.fps == pytest.approx(20.0)
    assert info.frame_count == 40
    assert info.decoded_frames == 40 and info.corrupted_frames == 0
    assert info.duration == pytest.approx(2.0)
    assert (info.width, info.height) == (64, 48)


def test_missing_video_raises(tmp_path):
    with pytest.raises(VideoUnavailableError):
        probe_video(tmp_path / "nope.mp4")


def test_reader_yields_requested_indices_in_order(synth_video):
    r = VideoReader(synth_video)
    got = list(r.read_frames([3, 0, 10, 10]))
    assert [i for i, _ in got] == [0, 3, 10]
    assert [frame_level(f) for _, f in got] == [0, 3, 10]


@pytest.mark.parametrize("seek", ["seek", "grab"])
def test_reader_segment_by_time(synth_video, seek):
    r = VideoReader(synth_video, start_time=0.5, end_time=1.0, seek=seek)  # frames 10..19
    assert (r.start_frame, r.end_frame, r.num_frames) == (10, 20, 10)
    frames = list(r.read_frames())
    assert len(frames) == 10
    assert frame_level(frames[0][1]) == 10 and frame_level(frames[-1][1]) == 19
    assert r.timestamp(0) == pytest.approx(0.5)


def test_reader_segment_past_end_is_clamped_and_reported(synth_video):
    r = VideoReader(synth_video, start_time=1.5, end_time=3.0)
    assert r.num_frames == 10
    assert r.truncated_tail_s == pytest.approx(1.0)


def test_reader_empty_segment_rejected(synth_video):
    with pytest.raises(ValueError):
        VideoReader(synth_video, start_time=5.0, end_time=6.0)


def test_reader_index_out_of_range(synth_video):
    with pytest.raises(IndexError):
        list(VideoReader(synth_video).read_frames([40]))


def test_image_folder_reader(synth_vsl_letters):
    r = ImageFolderReader(synth_vsl_letters / "A")
    assert r.num_frames == 6 and r.fps is None and r.timestamp(2) is None
    assert [frame_level(f) for _, f in r.read_frames([0, 5])] == [0, 10]


# ---------------------------------------------------------------- sampling / windows
def test_sample_indices_strategies():
    assert sample_indices(10, "all").tolist() == list(range(10))
    assert sample_indices(10, "stride", stride=3).tolist() == [0, 3, 6, 9]
    assert sample_indices(10, "uniform", num_samples=5).tolist() == [1, 3, 5, 7, 9]
    assert sample_indices(3, "uniform", num_samples=6).tolist() == [0, 0, 1, 1, 2, 2]   # short clip repeats
    assert sample_indices(48, "fps", src_fps=24, target_fps=12).tolist() == list(range(0, 48, 2))
    assert len(sample_indices(60, "fps", src_fps=30, target_fps=12)) == 24              # 2 s -> 24 steps
    assert len(sample_indices(1000, "all", max_frames=100)) == 100
    assert sample_indices(0, "all").size == 0
    with pytest.raises(ValueError):
        sample_indices(10, "fps", src_fps=None, target_fps=12)


def test_make_windows():
    assert make_windows(10, 4, 4) == [(0, 4), (4, 8), (8, 12)]               # padded tail
    assert make_windows(10, 4, 4, pad_last=False, min_last=3) == [(0, 4), (4, 8)]
    assert make_windows(3, 8, 4) == [(0, 8)]                                  # shorter than one window
    assert make_windows(16, 8, 4) == [(0, 8), (4, 12), (8, 16)]
    assert make_windows(0, 8, 4) == []


def test_time_helpers():
    assert time_to_frame_range(0.5, 1.0, 20, 40) == (10, 20)
    assert time_to_frame_range(None, 9.0, 20, 40) == (0, 40)
    assert interval_iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    assert interval_iou((0, 1), (2, 3)) == 0.0


def test_resize_letterbox_keeps_aspect():
    f = np.full((720, 1280, 3), 255, np.uint8)
    out = resize_frame(f, (224, 224), keep_aspect=True)
    assert out.shape == (224, 224, 3)
    assert out[0, 112].sum() == 0 and out[112, 112].sum() == 255 * 3   # black bars top, content centre


# ---------------------------------------------------------------- preprocessing pipeline
def _cfg(**kw):
    base = dict(sampling="all", image_size=(32, 32), window_size=8, window_stride=4, max_frames=None)
    base.update(kw)
    return PreprocessConfig(**base)


def test_process_shapes_and_normalization(synth_video):
    clip = VideoPreprocessor(_cfg(sampling="fps", target_fps=10)).process(VideoReader(synth_video))
    assert clip.frames.shape == (20, 3, 32, 32) and clip.frames.dtype == np.float32
    assert clip.valid_mask.all() and clip.effective_fps == 10
    assert clip.timestamps[1] == pytest.approx(0.1)
    assert clip.windows[0] == (0, 8) and len(clip.windows) == 4
    w, m = clip.window_array(3)
    assert w.shape == (8, 3, 32, 32) and m.sum() == 8   # windows (12,20)
    # normalization: gray 0 frame -> (0 - mean)/std
    assert clip.frames[0, 0, 16, 16] == pytest.approx((0 - 0.485) / 0.229, abs=1e-3)


def test_process_respects_max_frames(synth_video):
    clip = VideoPreprocessor(_cfg(max_frames=7)).process(VideoReader(synth_video))
    assert clip.length == 7 and clip.effective_fps == pytest.approx(20 * 7 / 40)


class _FlakySource:
    """Wraps a reader and drops given frame indices (simulates corrupted frames)."""

    def __init__(self, inner, bad):
        self.inner, self.bad = inner, set(bad)
        self.info, self.fps, self.num_frames = inner.info, inner.fps, inner.num_frames

    def timestamp(self, i):
        return self.inner.timestamp(i)

    def read_frames(self, indices=None):
        for i, f in self.inner.read_frames(indices):
            yield i, (None if i in self.bad else f)


@pytest.mark.parametrize("policy,expected_len", [("repeat_last", 10), ("zero", 10), ("drop", 7)])
def test_missing_frame_policies(synth_video, policy, expected_len):
    src = _FlakySource(VideoReader(synth_video, 0, 0.5), bad=[0, 4, 5])
    clip = VideoPreprocessor(_cfg(missing_frame_policy=policy, max_missing_ratio=0.5)).process(src)
    assert clip.length == expected_len and clip.n_missing == 3
    if policy != "drop":
        assert clip.valid_mask.tolist() == [False, True, True, True, False, False, True, True, True, True]
    if policy == "repeat_last":
        assert np.array_equal(clip.frames[0], clip.frames[1])   # leading gap back-filled
        assert np.array_equal(clip.frames[5], clip.frames[3])
    if policy == "zero":
        assert not clip.frames[4].any()


def test_too_many_missing_frames_raises(synth_video):
    src = _FlakySource(VideoReader(synth_video), bad=range(20))
    with pytest.raises(CorruptVideoError):
        VideoPreprocessor(_cfg(max_missing_ratio=0.2)).process(src)


def test_stream_windows_matches_process(synth_video):
    pre = VideoPreprocessor(_cfg(sampling="stride", stride=2))
    clip = pre.process(VideoReader(synth_video))
    streamed = list(pre.stream_windows(VideoReader(synth_video)))
    assert [(w.start, w.end) for w in streamed] == clip.windows
    for w in streamed:
        ref, ref_mask = clip.window_array(w.index)
        assert np.array_equal(w.frames, ref) and np.array_equal(w.mask, ref_mask)


def test_stream_windows_marks_missing_frames_invalid(synth_video):
    src = _FlakySource(VideoReader(synth_video), bad=[2])
    wins = list(VideoPreprocessor(_cfg()).stream_windows(src))
    assert wins[0].mask.tolist() == [True, True, False] + [True] * 5
    assert np.array_equal(wins[0].frames[2], wins[0].frames[1])   # repeat_last fill


def test_config_json_roundtrip(tmp_path):
    import json
    cfg = PreprocessConfig(target_fps=8, image_size=(112, 112))
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg.to_dict()))
    assert PreprocessConfig.from_json(p) == cfg


def test_default_config_file_loads():
    from conftest import REPO
    cfg = PreprocessConfig.from_json(REPO / "configs" / "preprocess_default.json")
    assert cfg.image_size == (224, 224) and cfg.missing_frame_policy == "repeat_last"


def test_variable_fps_sources_align_in_time(tmp_path):
    """24 fps and 30 fps sources resampled to 12 fps give equal-length sequences per second."""
    a = write_video(tmp_path / "a.avi", 48, 24.0)
    b = write_video(tmp_path / "b.avi", 60, 30.0)
    pre = VideoPreprocessor(_cfg(sampling="fps", target_fps=12))
    assert pre.process(VideoReader(a)).length == pre.process(VideoReader(b)).length == 24
