import pytest

from tlh.data.substitutes import SubstituteMap


def test_default_yaml_loads_and_validates():
    m = SubstituteMap.load()
    assert m.same_group("GOOGL", "GOOG")
    assert m.same_group("SPY", "IVV")           # presumed identical by default
    assert not m.same_group("SPY", "VTI")
    assert "VTI" in m.candidates_for("SPY")
    assert "IVV" not in m.candidates_for("SPY")  # never a same-group candidate
    assert m.candidates_for("AAPL")[:2] == ["MSFT", "XLK"]


def test_presumed_toggle():
    m = SubstituteMap.load(treat_presumed_as_identical=False)
    assert not m.same_group("SPY", "IVV")
    assert m.same_group("GOOGL", "GOOG")
    assert m.is_presumed("SPY")


def test_group_inheritance_of_candidates():
    m = SubstituteMap.from_dict({
        "presumed_identical_groups": [{"key": "g", "symbols": ["A", "B"]}],
        "substitutes": [{"for": ["A"], "candidates": ["C"]}],
    })
    assert m.candidates_for("B") == ["C"]


def test_validation_rejects_duplicate_membership():
    with pytest.raises(ValueError):
        SubstituteMap.from_dict({
            "identical_groups": [{"key": "x", "symbols": ["A", "B"]}],
            "presumed_identical_groups": [{"key": "y", "symbols": ["B", "C"]}],
        })


def test_validation_rejects_same_group_substitute():
    with pytest.raises(ValueError):
        SubstituteMap.from_dict({
            "presumed_identical_groups": [{"key": "x", "symbols": ["A", "B"]}],
            "substitutes": [{"for": ["A"], "candidates": ["B"]}],
        })


def test_to_substantially_identical_uses_resolver():
    m = SubstituteMap.from_dict({"identical_groups": [{"key": "x", "symbols": ["A", "B"]}]})
    si = m.to_substantially_identical({"A": 1, "B": 2, "C": 3}.get)
    assert si.same_group(1, 2) and not si.same_group(1, 3)
    assert "identical group" in m.explain_group("A")
