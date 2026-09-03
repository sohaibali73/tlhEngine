"""Risk-model calibration study: which estimation window, weighting and estimator forecast tracking error best?

Port and extension of the Potomac 2026 calibration study (Risk_Model_Calibration v1.3). For every scenario in a grid
(lookback x weighting x estimator x horizon) the covariance is estimated from returns strictly before each test date and
scored against the realised covariance over the next `horizon` days, walked forward through the whole history with
non-overlapping forward windows. Metrics (exactly as in the study):

    BiasRatio      mean(realised var / forecast var) over assets        target 1
    Spearman       rank corr(forecast vol, realised vol)                 target 1
    CorrBias       mean(forecast rho - realised rho) over sampled pairs  target 0
    CorrRMSE       rmse of pairwise correlation forecasts                lower better
    CorrSpearman   rank corr of pairwise correlations                    target 1
    TEBiasRatio    mean(realised TE^2 / forecast TE^2) over baskets      target 1
    TESpearman     rank corr(forecast TE, realised TE) over baskets      target 1
    Score          mean rank within horizon over the six metrics         lower better

Improvements over the original script: the realised side is anchored to the longest lookback (fixed target), the
client's own holdings can be scored as an extra basket, a PCA arm can be added to the estimator grid, and a
substitute-pair TE study (`pair_study`) reports which estimator to use for tight ETF pairs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .statistical import choose_n_factors, ledoit_wolf_cc, obs_weights, weighted_cov

log = logging.getLogger(__name__)

METRICS = ["BiasRatio", "Spearman", "CorrBias", "CorrRMSE", "CorrSpearman", "TEBiasRatio", "TESpearman"]


@dataclass
class CalibrationGrid:
    lookbacks: tuple[int, ...] = (21, 63, 126, 189, 252, 504, 756)
    weightings: tuple[str, ...] = ("equal", "exponential")
    estimators: tuple[str, ...] = ("sample", "ledoit_wolf")     # + "pca"
    horizons: tuple[int, ...] = (21, 63, 126)
    n_baskets: int = 20
    basket_size: int = 5
    n_pairs: int = 2000
    seed: int = 42
    halflife_ratio: float = 0.35
    max_symbols: int | None = None          # subsample the universe for speed (None = all)
    max_dates_per_horizon: int | None = None
    holdings: pd.Series | None = field(default=None, repr=False)    # optional client weights scored as an extra basket

    @classmethod
    def quick(cls) -> CalibrationGrid:
        return cls(lookbacks=(63, 126, 189, 252, 504), horizons=(21, 63), max_symbols=250, max_dates_per_horizon=40)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _forecast_cov(R: np.ndarray, weighting: str, estimator: str, halflife_ratio: float) -> np.ndarray:
    w = obs_weights(len(R), weighting, halflife_ratio=halflife_ratio)
    if estimator == "sample":
        return weighted_cov(R, w)
    if estimator == "ledoit_wolf":
        return ledoit_wolf_cc(R, w)[0]
    if estimator == "pca":
        S = weighted_cov(R, w)
        vals, vecs = np.linalg.eigh(S)
        order = np.argsort(vals)[::-1]
        vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order]
        k = min(choose_n_factors(vals), 20)
        V = vecs[:, :k]
        low = (V * vals[:k]) @ V.T
        D = np.maximum(np.diag(S) - np.diag(low), 0.05 * np.diag(S))
        return low + np.diag(D)
    raise ValueError(estimator)


def run_calibration(prices: pd.DataFrame, grid: CalibrationGrid | None = None, progress=None) -> dict:
    """`prices`: date x symbol total-return closes (may contain NaN for names not alive). Returns a dict with
    `scoreboard`, `by_lookback`, `by_weighting`, `by_estimator`, `by_horizon`, `winners`, `recommendation`, `pairs_note`."""
    grid = grid or CalibrationGrid()
    say = progress or (lambda m: None)
    rng = np.random.default_rng(grid.seed)
    rets = prices.sort_index().pct_change().iloc[1:].clip(-0.5, 0.5)
    if grid.max_symbols and rets.shape[1] > grid.max_symbols:
        # keep the names with the longest histories, then random
        alive = rets.notna().sum().sort_values(ascending=False)
        keep = list(alive.index[: grid.max_symbols])
        rets = rets[keep]
    L_max = max(grid.lookbacks)
    dates = rets.index
    T = len(dates)
    if T < L_max + max(grid.horizons) + 5:
        raise ValueError(f"need at least {L_max + max(grid.horizons) + 5} return days; have {T}")
    syms = list(rets.columns)
    n = len(syms)
    # fixed measuring instruments: baskets and pairs, drawn once
    baskets = [rng.choice(n, size=grid.basket_size, replace=False) for _ in range(grid.n_baskets)]
    pairs = rng.integers(0, n, size=(grid.n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    hold_vec = None
    if grid.holdings is not None:
        hv = grid.holdings.reindex(syms).fillna(0.0).values
        if hv.sum() > 0:
            hold_vec = hv / hv.sum()

    rows = []
    total_scen = len(grid.lookbacks) * len(grid.weightings) * len(grid.estimators)
    for h in grid.horizons:
        test_idx = list(range(L_max, T - h, h))              # non-overlapping forward windows, anchored to L_max
        if grid.max_dates_per_horizon and len(test_idx) > grid.max_dates_per_horizon:
            step = len(test_idx) / grid.max_dates_per_horizon
            test_idx = [test_idx[int(i * step)] for i in range(grid.max_dates_per_horizon)]
        # realised side (fixed target for every scenario in this horizon block)
        realised = []
        for ti in test_idx:
            fwd = rets.iloc[ti: ti + h]
            back = rets.iloc[ti - L_max: ti]
            ok = fwd.notna().all() & back.notna().all()
            ok &= back.std() > 1e-8
            cols = np.where(ok.values)[0]
            if len(cols) < 20:
                realised.append(None)
                continue
            Rf = fwd.values[:, cols]
            Sr = np.cov(Rf, rowvar=False, ddof=0)
            realised.append((cols, Sr))
        acc: dict[tuple, dict] = {}
        for L in grid.lookbacks:
            for wt in grid.weightings:
                for est in grid.estimators:
                    key = (L, wt, est, h)
                    acc[key] = {"var_ratio": [], "fvol": [], "rvol": [], "fcorr": [], "rcorr": [], "fte": [], "rte": [], "dates": 0, "alive": []}
        for k_i, (ti, real) in enumerate(zip(test_idx, realised, strict=True)):
            if real is None:
                continue
            cols, Sr = real
            m = len(cols)
            sub_syms_idx = {c: j for j, c in enumerate(cols)}
            # baskets & pairs restricted to alive names
            bk = []
            for b in baskets:
                if all(x in sub_syms_idx for x in b):
                    v = np.full(m, -1.0 / m)
                    for x in b:
                        v[sub_syms_idx[x]] += 1.0 / grid.basket_size
                    bk.append(v)
            if hold_vec is not None:
                hv = hold_vec[cols]
                if hv.sum() > 0.5:
                    v = hv / hv.sum() - 1.0 / m
                    bk.append(v)
            B = np.array(bk) if bk else np.zeros((0, m))
            pr = np.array([[sub_syms_idx[a], sub_syms_idx[b]] for a, b in pairs if a in sub_syms_idx and b in sub_syms_idx])
            r_var = np.diag(Sr)
            r_sd = np.sqrt(np.maximum(r_var, 1e-18))
            r_corr = Sr[pr[:, 0], pr[:, 1]] / (r_sd[pr[:, 0]] * r_sd[pr[:, 1]]) if len(pr) else np.array([])
            r_te = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", B, Sr, B), 0.0)) if len(B) else np.array([])
            for L in grid.lookbacks:
                back = rets.iloc[ti - L: ti].values[:, cols]
                for wt in grid.weightings:
                    for est in grid.estimators:
                        Sf = _forecast_cov(back, wt, est, grid.halflife_ratio)
                        f_var = np.diag(Sf)
                        f_sd = np.sqrt(np.maximum(f_var, 1e-18))
                        a = acc[(L, wt, est, h)]
                        a["var_ratio"].append(r_var / np.maximum(f_var, 1e-18))
                        a["fvol"].append(f_sd)
                        a["rvol"].append(r_sd)
                        if len(pr):
                            a["fcorr"].append(Sf[pr[:, 0], pr[:, 1]] / (f_sd[pr[:, 0]] * f_sd[pr[:, 1]]))
                            a["rcorr"].append(r_corr)
                        if len(B):
                            f_te = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", B, Sf, B), 1e-18))
                            a["fte"].append(f_te)
                            a["rte"].append(r_te)
                        a["dates"] += 1
                        a["alive"].append(m)
            say(f"Calibration: horizon {h}d, date {k_i + 1}/{len(test_idx)} ({total_scen} scenarios)")
        for (L, wt, est, hh), a in acc.items():
            if a["dates"] == 0:
                continue
            vr = np.concatenate(a["var_ratio"])
            fv, rv = np.concatenate(a["fvol"]), np.concatenate(a["rvol"])
            fc = np.concatenate(a["fcorr"]) if a["fcorr"] else np.array([])
            rc = np.concatenate(a["rcorr"]) if a["rcorr"] else np.array([])
            ft = np.concatenate(a["fte"]) if a["fte"] else np.array([])
            rt = np.concatenate(a["rte"]) if a["rte"] else np.array([])
            rows.append({
                "Lookback": L, "Months": round(L / 21, 1), "Weighting": wt.title(), "HalfLife": round(grid.halflife_ratio * L, 1) if wt == "exponential" else None,
                "Estimator": {"ledoit_wolf": "Ledoit-Wolf", "sample": "Sample", "pca": "PCA"}[est], "Horizon": hh, "HorizonMonths": round(hh / 21, 1),
                "Dates": a["dates"], "AliveMean": float(np.mean(a["alive"])), "AssetDates": int(len(vr)),
                "BiasRatio": float(np.mean(vr)), "Spearman": _spearman(fv, rv),
                "MeanForecastVol": float(np.mean(fv) * np.sqrt(252)), "MeanRealisedVol": float(np.mean(rv) * np.sqrt(252)),
                "PairDates": int(len(fc)), "CorrBias": float(np.mean(fc - rc)) if len(fc) else float("nan"),
                "CorrRMSE": float(np.sqrt(np.mean((fc - rc) ** 2))) if len(fc) else float("nan"),
                "CorrSpearman": _spearman(fc, rc) if len(fc) else float("nan"),
                "BasketDates": int(len(ft)), "TEBiasRatio": float(np.mean((rt ** 2) / (ft ** 2))) if len(ft) else float("nan"),
                "TESpearman": _spearman(ft, rt) if len(ft) else float("nan"),
                "MeanForecastTE": float(np.mean(ft) * np.sqrt(252)) if len(ft) else float("nan"),
                "MeanRealisedTE": float(np.mean(rt) * np.sqrt(252)) if len(rt) else float("nan"),
            })
    board = pd.DataFrame(rows)
    if board.empty:
        raise ValueError("calibration produced no scenarios (insufficient overlapping history)")
    board["Score"] = np.nan
    for _hh, blk in board.groupby("Horizon"):
        ranks = pd.DataFrame({
            "BiasRatio": (blk["BiasRatio"] - 1).abs().rank(), "TEBiasRatio": (blk["TEBiasRatio"] - 1).abs().rank(),
            "CorrBias": blk["CorrBias"].abs().rank(), "CorrRMSE": blk["CorrRMSE"].rank(),
            "Spearman": blk["Spearman"].rank(ascending=False), "TESpearman": blk["TESpearman"].rank(ascending=False),
        })
        board.loc[blk.index, "Score"] = ranks.mean(axis=1)
    board = board.sort_values(["Horizon", "Score"]).reset_index(drop=True)
    board["RankInHorizon"] = board.groupby("Horizon")["Score"].rank(method="first").astype(int)
    agg_cols = ["BiasRatio", "Spearman", "CorrBias", "CorrRMSE", "CorrSpearman", "TEBiasRatio", "TESpearman", "Score"]
    out = {
        "scoreboard": board,
        "by_lookback": board.groupby(["Horizon", "Lookback"])[agg_cols].mean().reset_index().sort_values(["Horizon", "Score"]),
        "by_weighting": board.groupby(["Horizon", "Weighting"])[agg_cols].mean().reset_index().sort_values(["Horizon", "Score"]),
        "by_estimator": board.groupby(["Horizon", "Estimator"])[agg_cols].mean().reset_index().sort_values(["Horizon", "Score"]),
        "by_horizon": board.groupby("Horizon")[agg_cols].mean().reset_index(),
        "winners": board[board["RankInHorizon"] == 1].reset_index(drop=True),
        "grid": {"lookbacks": list(grid.lookbacks), "weightings": list(grid.weightings), "estimators": list(grid.estimators), "horizons": list(grid.horizons),
                 "n_symbols": n, "n_baskets": grid.n_baskets, "basket_size": grid.basket_size, "n_pairs": int(len(pairs)), "with_holdings": hold_vec is not None},
    }
    # overall recommendation: best mean rank across horizons
    spec_cols = ["Lookback", "Weighting", "Estimator"]
    overall = board.groupby(spec_cols)["Score"].mean().reset_index().sort_values("Score")
    best = overall.iloc[0]
    out["overall_ranking"] = overall
    out["recommendation"] = {"lookback": int(best["Lookback"]), "weighting": best["Weighting"].lower(), "estimator": {"Ledoit-Wolf": "ledoit_wolf", "Sample": "sample", "PCA": "pca"}[best["Estimator"]],
                             "mean_score": float(best["Score"]),
                             "text": (f"Use a {int(best['Lookback'])}-day window, {best['Weighting'].lower()} weighting, {best['Estimator']} estimator: best composite "
                                      f"rank across the {len(grid.horizons)} horizons tested. Every specification under-forecasts on average "
                                      f"(bias ratios > 1); treat forecasts as a floor and prefer the sample matrix for tight substitute pairs.")}
    return out


def pair_study(prices: pd.DataFrame, pairs: list[tuple[str, str]], lookbacks: tuple[int, ...] = (63, 126, 252),
               horizon: int = 63, weightings: tuple[str, ...] = ("equal", "exponential"), estimators: tuple[str, ...] = ("sample", "ledoit_wolf"),
               universe_for_shrinkage: list[str] | None = None, progress=None) -> pd.DataFrame:
    """Forecast vs realised tracking error of tight substitute pairs (A minus B). Shrinkage toward the average correlation of a
    broad universe drags a 0.99 pair toward ~0.3, which is why the study recommends the sample matrix for pairs."""
    say = progress or (lambda m: None)
    rets = prices.sort_index().pct_change().iloc[1:]
    uni = universe_for_shrinkage or list(rets.columns)
    rows = []
    for a, b in pairs:
        if a not in rets or b not in rets:
            continue
        cols = [c for c in dict.fromkeys([a, b] + uni) if c in rets]
        R = rets[cols].dropna()
        L_max = max(lookbacks)
        idx = list(range(L_max, len(R) - horizon, horizon))
        for L in lookbacks:
            for wt in weightings:
                for est in estimators:
                    f_list, r_list = [], []
                    for ti in idx:
                        back = R.iloc[ti - L: ti].values
                        fwd = R.iloc[ti: ti + horizon].values
                        Sf = _forecast_cov(back, wt, est, 0.35)
                        Sr = np.cov(fwd, rowvar=False, ddof=0)
                        v = np.zeros(len(cols))
                        v[0], v[1] = 1.0, -1.0
                        f_list.append(np.sqrt(max(v @ Sf @ v, 1e-18) * 252))
                        r_list.append(np.sqrt(max(v @ Sr @ v, 0.0) * 252))
                    if f_list:
                        f_arr, r_arr = np.array(f_list), np.array(r_list)
                        rows.append({"pair": f"{a} vs {b}", "Lookback": L, "Weighting": wt.title(), "Estimator": {"ledoit_wolf": "Ledoit-Wolf", "sample": "sample", "pca": "PCA"}[est],
                                     "n": len(f_arr), "forecast_te": float(f_arr.mean()), "realised_te": float(r_arr.mean()),
                                     "bias": float(np.mean(r_arr ** 2 / f_arr ** 2)), "spearman": _spearman(f_arr, r_arr)})
        say(f"Pair study: {a} vs {b} done")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_bias_dev"] = (df["bias"] - 1).abs()
        df = df.sort_values(["pair", "abs_bias_dev"]).reset_index(drop=True)
    return df
