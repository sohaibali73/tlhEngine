"""Deep history store for research: every S&P 500 member since 1999 (including delisted names), split-adjusted closes,
cash dividends, point-in-time membership, GICS sectors, split-adjusted share counts and the S&P 500 total-return index.

Stored under var/research/store/ as numpy arrays (memory-mapped by worker processes: zero copy, instant open) plus a
small manifest. Norgate is only touched by `build_store`; everything else reads the store.

Caveats carried into every result:
  - Capitalisation weights are a proxy: today's shares outstanding scaled by the cumulative split factor (adjusted close
    times current shares). Buybacks and issuance are not reflected. The benchmark *return* is the real S&P 500 TR index.
  - Sectors are today's GICS classification (Norgate keeps the last known sector for delisted names).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WATCHLIST = "S&P 500 Current & Past"
INDEX_NAME = "S&P 500"
INDEX_TR = "$SPXTR"
INDEX_PX = "$SPX"
DEFAULT_START = "1999-01-04"


@dataclass
class ResearchStore:
    root: Path
    dates: pd.DatetimeIndex
    symbols: list[str]
    close: np.ndarray            # (T, N) split-adjusted close, NaN when not quoted
    dividend: np.ndarray         # (T, N) cash dividend per (adjusted) share on the ex-date, 0 otherwise
    member: np.ndarray           # (T, N) bool: in the S&P 500 that day
    shares: np.ndarray           # (N,) current shares outstanding (split-adjusted basis matches `close`)
    sector: np.ndarray           # (N,) object: GICS sector name or ""
    index_tr: np.ndarray         # (T,) total-return index level
    index_px: np.ndarray         # (T,) price index level
    manifest: dict

    # ------------------------------------------------------------------ derived
    @property
    def n_dates(self) -> int:
        return len(self.dates)

    def sym_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.symbols)}

    def cap_proxy(self, t: int) -> np.ndarray:
        """Capitalisation proxy at row t for members only (NaN elsewhere)."""
        c = self.close[t] * self.shares
        c = np.where(self.member[t] & np.isfinite(c) & (c > 0), c, np.nan)
        return c

    def bench_weights(self, t: int) -> np.ndarray:
        c = self.cap_proxy(t)
        w = np.nan_to_num(c, nan=0.0)
        s = w.sum()
        return w / s if s > 0 else w

    def returns(self) -> np.ndarray:
        """Daily total returns (T, N): price change plus dividend / previous close; NaN when not quoted."""
        prev = self.close[:-1]
        r = (self.close[1:] + self.dividend[1:]) / prev - 1.0
        out = np.full_like(self.close, np.nan)
        out[1:] = r
        return out

    def index_returns(self) -> np.ndarray:
        out = np.zeros(self.n_dates)
        out[1:] = self.index_tr[1:] / self.index_tr[:-1] - 1.0
        return np.nan_to_num(out)

    def date_pos(self, ts) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(ts)))

    def month_ends(self, start: int, end: int) -> list[int]:
        """Row positions of the last trading day of each month in [start, end)."""
        d = self.dates[start:end]
        if len(d) == 0:
            return []
        per = d.to_period("M")
        last = pd.Series(np.arange(start, end), index=per).groupby(level=0).last()
        return [int(x) for x in last.values]

    def summary(self) -> dict:
        return {"symbols": len(self.symbols), "dates": int(self.n_dates), "start": str(self.dates[0].date()), "end": str(self.dates[-1].date()),
                "members_now": int(self.member[-1].sum()), "members_first": int(self.member[0].sum()),
                "delisted": int(sum(1 for s in self.symbols if "-" in s)), **{k: v for k, v in self.manifest.items() if k in ("built", "watchlist", "seconds")}}


# ---------------------------------------------------------------------- build (Norgate)
def build_store(client, root: Path, start: str = DEFAULT_START, progress=None) -> ResearchStore:
    """Pull everything from Norgate and write the store. ~20-40 s for 1,900 symbols on a warm NDU."""
    say = progress or (lambda m: None)
    t0 = time.time()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    syms = list(client.watchlist_symbols(WATCHLIST))
    say(f"{len(syms)} current and past S&P 500 symbols; pulling split-adjusted prices from {start}…")
    px = client.price_panel(syms, start, adjustment="CAPITAL")
    if px.empty:
        raise RuntimeError("no price data returned from Norgate")
    say(f"{px.symbol.nunique()} symbols with prices; pulling index membership…")
    mem = client.index_constituents(syms, INDEX_NAME, start)
    say("pulling security metadata (sectors, shares)…")
    meta = client.securities_meta(syms).set_index("symbol")
    say("pulling the S&P 500 total-return and price indices…")
    idx_tr = client.price_history(INDEX_TR, start, adjustment="NONE")
    idx_px = client.price_history(INDEX_PX, start, adjustment="NONE")

    close = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    div = px.pivot_table(index="date", columns="symbol", values="dividend", aggfunc="sum").reindex(close.index).reindex(columns=close.columns).fillna(0.0)
    m = mem.pivot_table(index="date", columns="symbol", values="member", aggfunc="max").reindex(close.index).reindex(columns=close.columns)
    m = m.ffill().fillna(0.0).astype(bool)
    # a name is never a member on a day it has no close
    m = m & close.notna()
    symbols = list(close.columns)
    shares = pd.to_numeric(meta.get("shares_outstanding"), errors="coerce").reindex(symbols).fillna(np.nan).values.astype(float)
    sector = meta.get("gics_sector", pd.Series(index=symbols, dtype=object)).reindex(symbols).fillna("").astype(str).values
    tr = idx_tr.set_index("date")["close"].reindex(close.index).ffill().bfill().values.astype(float) if not idx_tr.empty else np.full(len(close), np.nan)
    pxi = idx_px.set_index("date")["close"].reindex(close.index).ffill().bfill().values.astype(float) if not idx_px.empty else np.full(len(close), np.nan)

    say("writing store…")
    np.save(root / "close.npy", close.values.astype(np.float64))
    np.save(root / "dividend.npy", div.values.astype(np.float64))
    np.save(root / "member.npy", m.values.astype(bool))
    np.save(root / "shares.npy", shares)
    np.save(root / "index_tr.npy", tr)
    np.save(root / "index_px.npy", pxi)
    manifest = {"built": pd.Timestamp.now().isoformat(timespec="seconds"), "watchlist": WATCHLIST, "index": INDEX_NAME, "start": start,
                "symbols": symbols, "dates": [d.strftime("%Y-%m-%d") for d in close.index], "sector": list(map(str, sector)),
                "seconds": round(time.time() - t0, 1), "n_symbols": len(symbols), "n_dates": int(len(close))}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    say(f"research store ready: {len(symbols)} symbols x {len(close)} days in {manifest['seconds']} s")
    return load_store(root)


def load_store(root: Path, mmap: bool = True) -> ResearchStore:
    root = Path(root)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    mode = "r" if mmap else None
    return ResearchStore(
        root=root, dates=pd.DatetimeIndex(pd.to_datetime(man["dates"])), symbols=list(man["symbols"]),
        close=np.load(root / "close.npy", mmap_mode=mode), dividend=np.load(root / "dividend.npy", mmap_mode=mode),
        member=np.load(root / "member.npy", mmap_mode=mode), shares=np.load(root / "shares.npy"),
        sector=np.array(man["sector"], dtype=object), index_tr=np.load(root / "index_tr.npy"), index_px=np.load(root / "index_px.npy"),
        manifest={k: v for k, v in man.items() if k not in ("symbols", "dates", "sector")})


def store_exists(root: Path) -> bool:
    root = Path(root)
    return (root / "manifest.json").exists() and (root / "close.npy").exists()


def store_from_frames(root: Path, close: pd.DataFrame, member: pd.DataFrame, shares: pd.Series, sector: pd.Series,
                      dividend: pd.DataFrame | None = None, index_tr: pd.Series | None = None) -> ResearchStore:
    """Build a store from in-memory frames (tests, synthetic markets, other data vendors)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    close = close.sort_index()
    symbols = list(close.columns)
    div = (dividend.reindex(close.index).reindex(columns=symbols).fillna(0.0) if dividend is not None else pd.DataFrame(0.0, index=close.index, columns=symbols))
    mem = member.reindex(close.index).reindex(columns=symbols).ffill().fillna(False).astype(bool) & close.notna()
    if index_tr is None:
        w = (close.values * shares.reindex(symbols).values[None, :])
        w = np.nan_to_num(np.where(mem.values, w, 0.0))
        r = np.nan_to_num((close.values[1:] + div.values[1:]) / close.values[:-1] - 1.0)
        wp = w[:-1] / np.clip(w[:-1].sum(axis=1, keepdims=True), 1e-12, None)
        lvl = np.concatenate([[100.0], 100.0 * np.cumprod(1 + (wp * r).sum(axis=1))])
        index_tr = pd.Series(lvl, index=close.index)
    np.save(root / "close.npy", close.values.astype(np.float64))
    np.save(root / "dividend.npy", div.values.astype(np.float64))
    np.save(root / "member.npy", mem.values.astype(bool))
    np.save(root / "shares.npy", shares.reindex(symbols).astype(float).values)
    np.save(root / "index_tr.npy", index_tr.reindex(close.index).ffill().bfill().values.astype(float))
    np.save(root / "index_px.npy", index_tr.reindex(close.index).ffill().bfill().values.astype(float))
    manifest = {"built": pd.Timestamp.now().isoformat(timespec="seconds"), "watchlist": "frames", "index": "cap-weighted proxy", "start": str(close.index[0].date()),
                "symbols": symbols, "dates": [d.strftime("%Y-%m-%d") for d in close.index], "sector": [str(sector.get(s, "")) for s in symbols],
                "seconds": 0.0, "n_symbols": len(symbols), "n_dates": int(len(close))}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return load_store(root)
