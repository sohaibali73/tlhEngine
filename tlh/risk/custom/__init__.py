"""Plugin folder for AI-authored (or hand-written) risk-model extensions.

Any `*.py` file dropped here is imported by `load_all()`; a module registers new style factors by adding to
`tlh.risk.factors.STYLE_DEFINITIONS`, e.g.

    from tlh.risk.factors import STYLE_DEFINITIONS, StyleDefinition, FactorInputs

    def dividend_yield(fi: FactorInputs):
        ...
    STYLE_DEFINITIONS["divyield"] = StyleDefinition("divyield", dividend_yield, "trailing dividend yield", True)

The co-pilot creates files here via `propose_change` (path prefix tlh/risk/custom/); the sandbox runs the risk-model
tests plus the wash-sale canary before promotion.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)
_LOADED: dict[str, float] = {}


def load_all(reload: bool = False) -> list[str]:
    folder = Path(__file__).parent
    loaded = []
    for p in sorted(folder.glob("*.py")):
        if p.name.startswith("_"):
            continue
        modname = f"{__name__}.{p.stem}"
        mtime = p.stat().st_mtime
        try:
            if modname in sys.modules and (reload or _LOADED.get(modname) != mtime):
                importlib.reload(sys.modules[modname])
            elif modname not in sys.modules:
                importlib.import_module(modname)
            _LOADED[modname] = mtime
            loaded.append(modname)
        except Exception as e:  # a broken plugin must never take the app down
            log.error("custom risk module %s failed to import: %s", p.name, e)
    return loaded
