"""Hand-computed WER cases."""
import pytest

from signbridge.metrics import EditCounts, corpus_wer, edit_counts


def test_identical():
    c = edit_counts("A B C", "A B C")
    assert c == EditCounts(0, 0, 0, 3)
    assert c.wer == 0.0


def test_one_substitution():
    assert edit_counts("A B C", "A X C") == EditCounts(1, 0, 0, 3)


def test_one_deletion():
    assert edit_counts("A B C", "A C") == EditCounts(0, 1, 0, 3)


def test_one_insertion():
    assert edit_counts("A B C", "A B X C") == EditCounts(0, 0, 1, 3)


def test_empty_hypothesis_is_all_deletions():
    c = edit_counts("A B C D", "")
    assert c == EditCounts(0, 4, 0, 4)
    assert c.wer == 1.0


def test_wer_can_exceed_one():
    c = edit_counts("A", "X Y Z")
    assert c.errors == 3  # 1 substitution + 2 insertions
    assert c.substitutions == 1 and c.insertions == 2
    assert c.wer == 3.0


def test_mixed_edits():
    # ref:  REGEN MORGEN NORD KALT
    # hyp:  REGEN HEUTE NORD SUED KALT -> MORGEN->HEUTE (S), insert SUED (I)
    c = edit_counts("REGEN MORGEN NORD KALT", "REGEN HEUTE NORD SUED KALT")
    assert c == EditCounts(1, 0, 1, 4)
    assert c.wer == pytest.approx(0.5)


def test_sequences_accepted():
    assert edit_counts(["A", "B"], ["A", "B"]).errors == 0


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        _ = edit_counts("", "A").wer


def test_corpus_wer_sums_edits_not_mean_of_rates():
    # sentence 1: 1 error / 1 ref token; sentence 2: 0 errors / 9 ref tokens
    c = corpus_wer(["A", "A B C D E F G H I"], ["X", "A B C D E F G H I"])
    assert c.errors == 1 and c.reference_length == 10
    assert c.wer == pytest.approx(0.1)  # the mean of per-sentence WER would be 0.5


def test_corpus_wer_length_mismatch():
    with pytest.raises(ValueError):
        corpus_wer(["A"], ["A", "B"])
