"""Loader for substitutes.yaml -> wash-sale groups + replacement candidates.

`SubstituteMap` works in symbol space and produces an assetid-keyed `SubstantiallyIdentical` for the wash
engine once a symbol->assetid resolver is supplied (Norgate or the securities table).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..tax.washsale import SubstantiallyIdentical

DEFAULT_PATH = Path(__file__).with_name("substitutes.yaml")
if not DEFAULT_PATH.exists():                     # frozen build
    import sys
    DEFAULT_PATH = Path(getattr(sys, "_MEIPASS", ".")) / "tlh" / "data" / "substitutes.yaml"


@dataclass
class SubstituteMap:
    identical: dict[str, set[str]] = field(default_factory=dict)         # key -> symbols
    presumed: dict[str, set[str]] = field(default_factory=dict)          # key -> symbols
    presumed_index: dict[str, str] = field(default_factory=dict)         # key -> index name
    substitutes: dict[str, list[str]] = field(default_factory=dict)      # symbol -> candidates (ordered)
    substitute_reason: dict[str, str] = field(default_factory=dict)
    stock_overrides: dict[str, list[str]] = field(default_factory=dict)
    stock_strategy: str = "same_gics_peer"
    stock_min_correlation: float = 0.55
    treat_presumed_as_identical: bool = True
    source_path: Path | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Path | str | None = None, treat_presumed_as_identical: bool = True) -> SubstituteMap:
        p = Path(path) if path else DEFAULT_PATH
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, treat_presumed_as_identical, source_path=p)

    @classmethod
    def from_dict(cls, raw: dict, treat_presumed_as_identical: bool = True, source_path: Path | None = None) -> SubstituteMap:
        m = cls(treat_presumed_as_identical=treat_presumed_as_identical, source_path=source_path)
        for g in raw.get("identical_groups", []) or []:
            m.identical[g["key"]] = {_u(s) for s in g.get("symbols", [])}
        for g in raw.get("presumed_identical_groups", []) or []:
            m.presumed[g["key"]] = {_u(s) for s in g.get("symbols", [])}
            if g.get("index"):
                m.presumed_index[g["key"]] = g["index"]
        for row in raw.get("substitutes", []) or []:
            cands = [_u(c) for c in row.get("candidates", [])]
            for s in row.get("for", []):
                s = _u(s)
                m.substitutes.setdefault(s, [])
                for c in cands:
                    if c not in m.substitutes[s] and c != s:
                        m.substitutes[s].append(c)
                if row.get("reason"):
                    m.substitute_reason[s] = row["reason"]
        ss = raw.get("stock_substitutes", {}) or {}
        m.stock_strategy = ss.get("strategy", m.stock_strategy)
        m.stock_min_correlation = float(ss.get("min_correlation", m.stock_min_correlation))
        m.stock_overrides = {_u(k): [_u(x) for x in v] for k, v in (ss.get("overrides", {}) or {}).items()}
        m._validate()
        return m

    def _validate(self) -> None:
        # a symbol may appear in at most one identical/presumed group
        seen: dict[str, str] = {}
        for key, syms in list(self.identical.items()) + list(self.presumed.items()):
            for s in syms:
                if s in seen and seen[s] != key:
                    raise ValueError(f"{s} appears in groups {seen[s]} and {key}")
                seen[s] = key
        # substitutes must never point at something in the same group
        for s, cands in self.substitutes.items():
            g = self.group_key(s)
            for c in cands:
                if self.group_key(c) == g:
                    raise ValueError(f"substitute {c} for {s} is in the same substantially-identical group {g}")

    # ------------------------------------------------------------------ groups
    def group_key(self, symbol: str) -> str:
        s = _u(symbol)
        for key, syms in self.identical.items():
            if s in syms:
                return f"ident:{key}"
        if self.treat_presumed_as_identical:
            for key, syms in self.presumed.items():
                if s in syms:
                    return f"presumed:{key}"
        return f"sym:{s}"

    def group_members(self, symbol: str) -> set[str]:
        key = self.group_key(symbol)
        if key.startswith("ident:"):
            return set(self.identical[key[6:]])
        if key.startswith("presumed:"):
            return set(self.presumed[key[9:]])
        return {_u(symbol)}

    def same_group(self, a: str, b: str) -> bool:
        return self.group_key(a) == self.group_key(b)

    def is_presumed(self, symbol: str) -> bool:
        s = _u(symbol)
        return any(s in syms for syms in self.presumed.values())

    def explain_group(self, symbol: str) -> str:
        key = self.group_key(symbol)
        if key.startswith("ident:"):
            return f"{symbol} is in identical group '{key[6:]}' with {sorted(self.identical[key[6:]] - {_u(symbol)})}."
        if key.startswith("presumed:"):
            k = key[9:]
            idx = self.presumed_index.get(k, "the same index")
            return (f"{symbol} is presumed substantially identical to {sorted(self.presumed[k] - {_u(symbol)})} "
                    f"(all track {idx}). Toggle in Settings if you disagree.")
        return f"{symbol} has no substantially-identical peers in the mapping."

    def to_substantially_identical(self, resolve: Callable[[str], int | None]) -> SubstantiallyIdentical:
        """Build the assetid-keyed mapping the wash engine uses."""
        mapping: dict[int, str] = {}
        for key, syms in self.identical.items():
            for s in syms:
                aid = resolve(s)
                if aid is not None:
                    mapping[aid] = f"ident:{key}"
        if self.treat_presumed_as_identical:
            for key, syms in self.presumed.items():
                for s in syms:
                    aid = resolve(s)
                    if aid is not None:
                        mapping[aid] = f"presumed:{key}"
        return SubstantiallyIdentical(mapping)

    # ------------------------------------------------------------------ replacements
    def candidates_for(self, symbol: str) -> list[str]:
        s = _u(symbol)
        out: list[str] = []
        for c in self.substitutes.get(s, []) + self.stock_overrides.get(s, []):
            if c not in out and not self.same_group(s, c):
                out.append(c)
        # inherit candidates from group-mates (e.g. SPY inherits IVV's list)
        for mate in self.group_members(s) - {s}:
            for c in self.substitutes.get(mate, []):
                if c not in out and not self.same_group(s, c):
                    out.append(c)
        return out

    def all_symbols(self) -> set[str]:
        out: set[str] = set()
        for syms in self.identical.values():
            out |= syms
        for syms in self.presumed.values():
            out |= syms
        for s, cands in self.substitutes.items():
            out.add(s)
            out |= set(cands)
        for s, cands in self.stock_overrides.items():
            out.add(s)
            out |= set(cands)
        return out


def _u(s: str) -> str:
    return str(s).strip().upper()
