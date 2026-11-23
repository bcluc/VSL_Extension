"""Isolated-sign demo: phrase composer, feature invariants, classifier shapes (no dataset needed)."""
import numpy as np
import torch

from signbridge.signs.features import RawSequence, active_span, feature_dim, mirror, sequence_features
from signbridge.signs.model import SignClassifier
from signbridge.signs.phrases import base_gloss, compose


def test_variants_collapse():
    assert base_gloss("WHAT1") == "WHAT" and base_gloss("DEAF2") == "DEAF" and base_gloss("HELLO") == "HELLO"


def test_compose_longest_match():
    c = compose(["HELLO", "NICE", "MEET", "YOU"])
    assert c.english == "Hello. Nice to meet you."
    assert c.vietnamese == "Xin chào. Rất vui được gặp bạn."
    assert c.matched


def test_compose_variants_and_question():
    assert compose(["HOW2", "YOU"]).english == "How are you?"
    assert compose(["YOUR", "NAME", "WHAT1"]).english == "What is your name?"


def test_compose_unknown_falls_back_to_glosses():
    c = compose(["YOU", "SIGN"])
    assert not c.matched
    assert c.english == "[you] [sign]"


def test_compose_empty():
    assert compose([]).english == ""


def _seq(n=20, seed=0):
    rng = np.random.default_rng(seed)
    pose = rng.uniform(0.2, 0.8, (n, 33, 4)).astype(np.float32)
    hands = rng.uniform(0.2, 0.8, (n, 2, 21, 3)).astype(np.float32)
    hp = np.zeros((n, 2), bool)
    hp[5:15, 1] = True
    hp[8:12, 0] = True
    hands *= hp[..., None, None]
    return RawSequence(pose, hands, hp, np.ones(n, bool))


def test_active_span_and_shape():
    s = _seq()
    assert active_span(s.hand_present, margin=2) == (3, 17)
    f = sequence_features(s, t=16)
    assert f.shape == (16, feature_dim(16)) and np.isfinite(f).all()


def test_features_invariant_to_translation_and_scale():
    s = _seq()
    moved = RawSequence(s.pose.copy(), s.hands.copy(), s.hand_present, s.pose_present)
    moved.pose[..., :2] = moved.pose[..., :2] * 0.5 + 0.1
    moved.hands[..., :2] = np.where(s.hand_present[..., None, None], moved.hands[..., :2] * 0.5 + 0.1, 0)
    a, b = sequence_features(s), sequence_features(moved)
    # body-relative coordinates and motion are unchanged; hand shape uses z too, which we did not scale
    n_xy = 13 * 2 + 2 * 21 * 2
    np.testing.assert_allclose(a[:, :n_xy], b[:, :n_xy], atol=1e-4)


def test_mirror_is_involution():
    s = _seq()
    m = mirror(mirror(s))
    np.testing.assert_allclose(m.hands, s.hands, atol=1e-6)
    assert (m.hand_present == s.hand_present).all()


class _Fixed(torch.nn.Module):
    """Stub model that always predicts class ``k`` with high confidence."""

    def __init__(self, n, k):
        super().__init__()
        self.n, self.k = n, k

    def forward(self, x):
        out = torch.full((x.shape[0], self.n), -5.0)
        out[:, self.k] = 5.0
        return out


def _frame(active: bool):
    pose = np.zeros((33, 4), np.float32)
    pose[11, :2] = (0.6, 0.4)
    pose[12, :2] = (0.4, 0.4)
    hands = np.zeros((2, 21, 3), np.float32)
    present = np.zeros(2, bool)
    if active:
        hands[1, :, :2] = (0.45, 0.3)  # wrist above the shoulders
        present[1] = True
    return pose, hands, present, True


def test_live_session_segments_signs_and_sentences():
    from signbridge.signs.live import LiveConfig, SentenceEvent, SignEvent, SignSession
    classes = ["BYE", "HELLO"]
    s = SignSession(_Fixed(2, 1), classes, t=8, cfg=LiveConfig())
    events, fps = [], 30.0
    timeline = [False] * 10 + [True] * 20 + [False] * 20 + [True] * 20 + [False] * 90
    for i, a in enumerate(timeline):
        events += s.push(i / fps, *_frame(a))
    signs = [e for e in events if isinstance(e, SignEvent)]
    sents = [e for e in events if isinstance(e, SentenceEvent)]
    assert [e.gloss for e in signs] == ["HELLO", "HELLO"] and all(e.accepted for e in signs)
    assert len(sents) == 1 and sents[0].glosses == ["HELLO", "HELLO"]


def test_live_session_rejects_low_confidence_and_resting_hands():
    from signbridge.signs.live import LiveConfig, SignEvent, SignSession, is_active
    cfg = LiveConfig(min_confidence=0.99999)  # the stub's confidence is ~0.9999
    pose, hands, present, ok = _frame(True)
    hands[1, :, 1] = 0.95  # hand visible but resting low (below the rest line)
    assert not is_active(pose, hands, present, ok, cfg)
    s = SignSession(_Fixed(3, 0), ["A", "B", "C"], t=8, cfg=cfg)
    ev = []
    for i, a in enumerate([True] * 20 + [False] * 30):
        ev += s.push(i / 30, *_frame(a))
    signs = [e for e in ev if isinstance(e, SignEvent)]
    assert len(signs) == 1 and not signs[0].accepted and s.pending_glosses == []


def test_classifier_shapes():
    m = SignClassifier(feature_dim(32), 29)
    out = m(torch.zeros(3, 32, feature_dim(32)))
    assert out.shape == (3, 29)


def test_extractor_rejects_unknown_mode():
    import pytest
    from signbridge.signs.features import LandmarkExtractor
    with pytest.raises(ValueError):
        LandmarkExtractor("missing_pose.task", "missing_hand.task", mode="stream")


def test_latest_frame_drops_stale_frames():
    """The live capture thread must hand out the newest frame, never a backlog of old ones."""
    import threading

    from signbridge.live_io import LatestFrame

    release = threading.Event()

    class FakeCap:
        def __init__(self):
            self.i = 0

        def read(self):
            if self.i == 5:
                release.wait(2.0)  # hold the camera after 5 frames
            if self.i >= 6:
                return False, None
            self.i += 1
            return True, np.full((2, 2, 3), self.i, np.uint8)

    g = LatestFrame(FakeCap())
    g.start()
    import time
    t_end = time.time() + 2
    while g.seq < 5 and time.time() < t_end:  # the camera produced 5 frames while we were busy
        time.sleep(0.01)
    frame, _, seq = g.get(0)
    assert int(frame[0, 0, 0]) == 5  # frames 2..4 were skipped, not queued
    release.set()
    frame, _, seq = g.get(seq)
    assert int(frame[0, 0, 0]) == 6
    assert g.get(seq, timeout=1.0)[0] is None  # camera stopped
    g.stop()


def test_ensemble_averages_probabilities():
    from signbridge.signs.model import EnsembleClassifier
    torch.manual_seed(0)
    a, b = SignClassifier(feature_dim(), 5).eval(), SignClassifier(feature_dim(), 5).eval()
    x = torch.randn(3, 32, feature_dim())
    with torch.no_grad():
        want = (torch.softmax(a(x), 1) + torch.softmax(b(x), 1)) / 2
        got = torch.softmax(EnsembleClassifier([a, b])(x), 1)
    assert torch.allclose(got, want, atol=1e-6)


def test_debounce_ignores_single_frame_flicker():
    from signbridge.signs.live import LiveConfig, SignSession
    pose = np.zeros((33, 4), np.float32)
    hands = np.zeros((2, 21, 3), np.float32)
    on, off = np.array([True, False]), np.array([False, False])
    s = SignSession(torch.nn.Identity(), ["A"], 32, LiveConfig(active_window=3, active_min=2, start_frames=1))
    for i, pres in enumerate([on, off, off, on, off, off]):  # isolated flickers, never 2 of 3
        s.push(i / 30, pose, hands, pres, False)
        assert not s.in_sign
    s.push(6 / 30, pose, hands, on, False)
    s.push(7 / 30, pose, hands, on, False)  # now 2 of the last 3 frames are active
    assert s.in_sign
