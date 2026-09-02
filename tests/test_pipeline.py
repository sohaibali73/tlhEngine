import numpy as np
import pandas as pd

from tlh.optim.pipeline import (
    EXAMPLES,
    NODE_TYPES,
    Pipeline,
    apply_filter,
    defaults,
    new_node,
    ordered,
    parse_kv,
    rank_keep,
    spec_from_nodes,
    validate,
)


def test_examples_validate_and_roundtrip():
    for name, p in EXAMPLES.items():
        assert validate(p) == [], name
        q = Pipeline.from_json(p.to_json())
        assert [n.type for n in ordered(q)] == [n.type for n in ordered(p)]
        assert q.name == p.name


def test_defaults_cover_every_param():
    for t, spec in NODE_TYPES.items():
        d = defaults(t)
        assert set(d) == {p["name"] for p in spec["params"]}
        n = new_node(t, "x")
        assert n.params == d


def test_validation_rules():
    assert validate(Pipeline("e"))[0].startswith("pipeline is empty")
    p = Pipeline("no universe", [new_node("construct", "c", 0)])
    assert any("Universe" in e for e in validate(p))
    p = Pipeline("bad order", [new_node("construct", "c", 0), new_node("universe", "u", 200)])
    assert any("left-most" in e for e in validate(p))
    p = Pipeline("filter after construct", [new_node("universe", "u", 0), new_node("construct", "c", 200), new_node("filter", "f", 400)])
    assert any("before Construction" in e for e in validate(p))
    p = Pipeline("transition alone", [new_node("universe", "u", 0), new_node("transition", "t", 200)])
    assert any("needs a Construction" in e for e in validate(p))
    ok = Pipeline("ok", [new_node("universe", "u", 0), new_node("filter", "f", 200), new_node("construct", "c", 400), new_node("output", "o", 600)])
    assert validate(ok) == []


def test_ordered_uses_x_then_stage_hint():
    p = Pipeline("o", [new_node("output", "o", 500), new_node("universe", "u", 0), new_node("construct", "c", 250), new_node("benchmark", "b", 250)])
    assert [n.type for n in ordered(p)] == ["universe", "benchmark", "construct", "output"]


def test_apply_filter_and_rank():
    syms = ["A", "B", "C", "D"]
    sec = pd.DataFrame({"gics_sector": ["Energy", "Information Technology", "Health Care", "Energy"]}, index=syms)
    mc = pd.Series([500, 50000, 20000, 100], index=syms)
    ret = pd.Series([-0.2, 0.3, 0.1, 0.05], index=syms)
    vol = pd.Series([0.5, 0.3, 0.2, 0.9], index=syms)
    out = apply_filter(syms, sec, mc, ret, vol, {"min_mktcap_musd": 1000, "sectors_exclude": "energy", "min_ret_1y": 0.0, "max_vol_1y": 0.4})
    assert out == ["B", "C"]
    out2 = apply_filter(syms, sec, mc, ret, vol, {"sectors_include": "energy", "exclude_symbols": "D"})
    assert out2 == ["A"]
    sig = pd.DataFrame({"momentum": [1.0, -1.0, 0.5, 0.0], "quality": [0.0, 3.0, 1.0, -1.0]}, index=syms)
    kept, score = rank_keep(syms, sig, {"signal_weights": "momentum=1, quality=1", "top_n": 2})
    assert kept == ["B", "C"]                                 # scores B=2.0, C=1.5, A=1.0, D=-1.0
    kept_low, _ = rank_keep(syms, sig, {"signal_weights": "momentum=1", "top_n": 1, "ascending": True})
    assert kept_low == ["B"]


def test_spec_from_nodes():
    spec = spec_from_nodes({"strategy": "mean_variance", "n_max": 30, "max_weight": 0.05, "sector_band": 0, "signal_weights": "momentum=1, value=0.5",
                            "ic": 0.1, "risk_aversion": 3, "views": '[{"assets": {"AAPL": 1}, "return": 0.05, "confidence": 0.5}]'},
                           {"gain_budget": 0.02, "turnover_max": 0.3})
    assert spec.kind == "mean_variance" and spec.n_max == 30 and spec.sector_band is None
    assert spec.signal_weights == {"momentum": 1.0, "value": 0.5} and spec.ic == 0.1
    assert spec.views[0]["assets"] == {"AAPL": 1} and spec.gain_budget == 0.02 and spec.turnover_max == 0.3
    assert parse_kv("a=1, b=x, c=2.5") == {"a": 1.0, "c": 2.5}
    assert np.isclose(spec.risk_aversion, 3.0)
