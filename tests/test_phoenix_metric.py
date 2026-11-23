"""PHOENIX gloss normalisation (hand cases) and agreement with the official sclite output."""
import importlib.util
from pathlib import Path

import pytest

from signbridge.metrics import EditCounts, edit_counts
from signbridge.metrics.phoenix import normalize_hypothesis, normalize_reference

REPO = Path(__file__).resolve().parents[1]
EVAL = (REPO.parent / "datasets" / "phoenix2014T" / "PHOENIX-2014-T-release-v3" / "PHOENIX-2014-T"
        / "evaluation" / "sign-recognition")


def test_hypothesis_drops_meta_and_prefixes():
    hyp = ["__ON__", "loc-NORDRAUM", "REGEN-PLUSPLUS", "cl-KOMMEN", "__EMOTION__", "KOMMEN", "__OFF__"]
    assert normalize_hypothesis(hyp) == ["NORD", "REGEN", "KOMMEN"]  # repeated KOMMEN collapsed


def test_hypothesis_keeps_bare_raum():
    assert normalize_hypothesis(["RAUM"]) == ["RAUM"]


def test_reference_compounds_spelled_letters():
    assert normalize_reference("HEUTE A S NACHT") == ["HEUTE", "A+S", "NACHT"]


def test_reference_wie_aussehen_and_repeats():
    assert normalize_reference("WIE AUSSEHEN MORGEN MORGEN") == ["WIE-AUSSEHEN", "MORGEN"]


def test_reference_meta_tokens_removed():
    assert normalize_reference("__ON__ REGEN loc-SUEDRAUM __OFF__") == ["REGEN", "SUED"]


def test_unit_weights_option():
    assert edit_counts("A B", "B A", weights=(1, 1, 1)).errors == 2


@pytest.mark.phoenix
def test_matches_official_sclite_output():
    if not EVAL.is_dir():
        pytest.skip("PHOENIX-2014-T evaluation folder not downloaded")
    spec = importlib.util.spec_from_file_location("check_phoenix_wer", REPO / "scripts" / "check_phoenix_wer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from signbridge.metrics.phoenix import phoenix_wer, read_ctm, read_stm

    for split in ("dev", "test"):
        refs = read_stm(EVAL / f"PHOENIX-2014-T-groundtruth-{split}.stm")
        c: EditCounts = phoenix_wer(refs, read_ctm(EVAL / f"example-hypothesis-{split}.ctm"))
        sc = mod.sclite_sum(EVAL / f"out.example-hypothesis-{split}.ctm.sys")
        assert c.reference_length == sc["words"]
        for ours, key in ((c.substitutions, "sub"), (c.deletions, "del"), (c.insertions, "ins"), (c.errors, "err")):
            assert round(100 * ours / c.reference_length, 1) == pytest.approx(sc[key], abs=0.05)
