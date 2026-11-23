"""signbridge.letters: features, label composition, model checkpoint, typing logic, recordings.

All synthetic: no recorded letters exist in the repository.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from signbridge.letters.features import FEATURE_DIM, augment, letter_features
from signbridge.letters.labels import DEFAULT_LABELS, append_letter, display
from signbridge.letters.model import LetterMLP, load_letter_model, save_letter_model
from signbridge.letters.recognizer import LetterPrediction, LetterTyper
from signbridge.letters.recordings import count_takes, load_takes, save_take

REPO = Path(__file__).resolve().parents[1]


def _hand(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.3, 0.7, (21, 3)).astype(np.float32)
    p[:, 2] *= 0.1
    return p


def test_feature_shape_and_finite():
    f = letter_features(np.stack([_hand(0), _hand(1)]), [False, True], [16 / 9, 16 / 9])
    assert f.shape == (2, FEATURE_DIM) and np.isfinite(f).all()


def test_features_ignore_position_and_size():
    h = _hand(2)
    moved = h.copy()
    moved[:, :2] = (moved[:, :2] - 0.5) * 0.6 + 0.4   # smaller and shifted
    moved[:, 2] *= 0.6
    a = letter_features(h, False, 1.0)
    b = letter_features(moved, False, 1.0)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_left_hand_is_mirrored_onto_right():
    h = _hand(3)
    mirrored = h.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    np.testing.assert_allclose(letter_features(h, False, 1.0), letter_features(mirrored, True, 1.0), atol=1e-5)


def test_augment_keeps_shape_and_changes_points():
    x = np.stack([_hand(i) for i in range(4)])
    y = augment(x, np.random.default_rng(0))
    assert y.shape == x.shape and not np.allclose(x, y)


def test_default_labels_unique_ascii_ids():
    ids = [k for k, _ in DEFAULT_LABELS]
    assert len(ids) == len(set(ids)) and all(k.isascii() for k in ids)
    assert display("DD") == "Đ" and display("ZZ") == "ZZ"


@pytest.mark.parametrize("seq, expected", [
    (["A", "MU"], "Â"), (["O", "RAU"], "Ơ"), (["U", "RAU"], "Ư"), (["E", "MU"], "Ê"),
    (["MU"], ""), (["B", "RAU"], "B"), (["DD", "O"], "ĐO"),
])
def test_append_letter_modifiers(seq, expected):
    t = ""
    for s in seq:
        t = append_letter(t, s)
    assert t == expected


def test_checkpoint_round_trip(tmp_path):
    labels = ["A", "B", "C"]
    m = LetterMLP(len(labels), hidden=32).eval()
    m.set_normalisation(np.random.default_rng(0).normal(size=(10, FEATURE_DIM)).astype(np.float32))
    path = tmp_path / "m.pt"
    save_letter_model(path, m, labels, 32, {"note": "test"})
    m2, labels2, meta = load_letter_model(path)
    x = torch.randn(4, FEATURE_DIM)
    assert labels2 == labels and meta["note"] == "test"
    torch.testing.assert_close(m(x), m2(x))


def test_missing_checkpoint_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="record_letters"):
        load_letter_model(tmp_path / "none.pt")


def test_foreign_checkpoint_is_refused(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        load_letter_model(tmp_path / "x.pt")


def _pred(probs):
    return LetterPrediction(hand=object(), probs=np.array(probs, np.float32), labels=["A", "B"])


def test_typer_types_after_hold_and_not_twice():
    ty = LetterTyper(["A", "B"], window=2, threshold=0.6, hold_s=0.5)
    typed = [ty.push(_pred([0.9, 0.1]), t) for t in np.arange(0, 1.5, 0.1)]
    assert typed.count("A") == 1 and ty.text == "A"
    ty.push(LetterPrediction(None, labels=["A", "B"]), 1.6)      # hand leaves
    for t in np.arange(1.7, 2.6, 0.1):
        ty.push(_pred([0.9, 0.1]), t)
    assert ty.text == "AA"


def test_typer_ignores_low_confidence():
    ty = LetterTyper(["A", "B"], window=2, threshold=0.6, hold_s=0.2)
    for t in np.arange(0, 2, 0.1):
        ty.push(_pred([0.55, 0.45]), t)
    assert ty.text == ""


def test_recordings_round_trip(tmp_path):
    save_take(tmp_path, "s01", "A", [_hand(0), _hand(1)], [False, False], 16 / 9, 20.0)
    save_take(tmp_path, "s01", "B", [_hand(2)], [True], 16 / 9, 20.0)
    takes = load_takes(tmp_path)
    assert [(t.signer, t.label, len(t)) for t in takes] == [("s01", "A", 2), ("s01", "B", 1)]
    assert count_takes(tmp_path, "s01") == {"A": 1, "B": 1}
    assert load_takes(tmp_path, ["B"])[0].is_left.tolist() == [True]


def test_split_holds_out_whole_takes(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    from train_letter_classifier import split_takes
    for lab in ("A", "B"):
        for _ in range(4):
            save_take(tmp_path, "s01", lab, [_hand(0)] * 3, [False] * 3, 1.0, 20.0)
    sp = split_takes(load_takes(tmp_path), None)
    assert [len(sp[k]) for k in ("train", "val", "test")] == [4, 2, 2]
    names = {k: {t.path for t in v} for k, v in sp.items()}
    assert not (names["train"] & names["test"]) and not (names["val"] & names["test"])
    with pytest.raises(SystemExit):
        split_takes(load_takes(tmp_path)[:2], None)   # only 2 takes of A


def test_classifier_learns_separable_synthetic_letters():
    """Sanity check of features + MLP on two clearly different synthetic hand shapes."""
    rng = np.random.default_rng(0)
    base = [_hand(10), _hand(11)]
    x = np.concatenate([augment(np.repeat(b[None], 200, 0), rng) for b in base])
    y = torch.tensor([0] * 200 + [1] * 200)
    feats = torch.from_numpy(letter_features(x, np.zeros(400, bool), np.ones(400, np.float32)))
    m = LetterMLP(2, hidden=32)
    m.set_normalisation(feats.numpy())
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(60):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(feats), y).backward()
        opt.step()
    assert (m.eval()(feats).argmax(1) == y).float().mean() > 0.95
