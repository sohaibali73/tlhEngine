"""Application service for the TLH research laboratory (no Qt): store management, study design, execution, results."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..research import grid
from ..research.data import build_store, load_store, store_exists
from ..research.report import excel_report, markdown_report, study_from_json
from ..research.spec import APPROACHES, ResearchSpec, StudySpec
from .context import AppContext

log = logging.getLogger(__name__)


class ResearchService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.root = Path(ctx.settings.var_dir) / "research"
        self.store_root = self.root / "store"
        self.studies_root = self.root / "studies"
        self._store = None

    # ------------------------------------------------------------------ store
    def store_status(self) -> dict:
        if not store_exists(self.store_root):
            return {"ready": False, "path": str(self.store_root)}
        st = self.store()
        return {"ready": True, "path": str(self.store_root), **st.summary(), "last_year": int(st.dates[-1].year)}

    def store(self):
        if self._store is None:
            self._store = load_store(self.store_root)
        return self._store

    def build_store(self, progress=None):
        self.ctx.norgate.require()
        self._store = build_store(self.ctx.norgate, self.store_root, progress=progress)
        self.ctx.db.audit("user", "research.build_store", None, symbols=len(self._store.symbols))
        return self._store.summary()

    # ------------------------------------------------------------------ studies
    def default_study(self, name: str = "MVP", quick: bool = False) -> StudySpec:
        base = ResearchSpec(horizon_years=10, account_size=500_000, basket_size=150, trigger=0.0025, approach="optimizer", te_limit=0.02, sector_band=0.02)
        return StudySpec(name=name, base=base, sweeps=["account_size", "basket_size", "trigger", "approach"], horizons=[10],
                         first_start_year=2000, every_n_years=3 if quick else 1)

    def estimate(self, study: StudySpec) -> dict:
        return grid.estimate(study, self.store().dates[-1].year)

    def run_study(self, study: StudySpec, progress=None, cancel=None, workers: int | None = None) -> dict:
        if not store_exists(self.store_root):
            raise RuntimeError("build the research store first (Norgate Data Updater must be running)")
        out = grid.run_study(study, self.store_root, self.studies_root, self.store().dates[-1].year, workers=workers, progress=progress, cancel=cancel)
        res, _ = grid.load_results(out)
        self.ctx.db.audit("user", "research.run_study", study.name, runs=int(len(res)))
        return {"study": study.name, "folder": str(out), "runs": int(len(res)), "failed": int(res["error"].notna().sum()) if "error" in res else 0}

    def run_single(self, spec: ResearchSpec, progress=None):
        from ..research.engine import run_window
        return run_window(self.store(), spec, progress=progress)

    def list_studies(self) -> list[dict]:
        out = []
        if not self.studies_root.exists():
            return out
        for d in sorted(self.studies_root.iterdir()):
            if (d / "study.json").exists():
                res = pd.read_parquet(d / "results.parquet") if (d / "results.parquet").exists() else pd.DataFrame()
                st = json.loads((d / "study.json").read_text(encoding="utf-8"))
                out.append({"name": st.get("name", d.name), "folder": str(d), "runs": int(len(res)), "sweeps": st.get("sweeps", []),
                            "horizons": st.get("horizons", []), "modified": pd.Timestamp((d / "study.json").stat().st_mtime, unit="s").isoformat(timespec="minutes")})
        return out

    def _dir(self, name: str) -> Path:
        d = self.studies_root / grid._slug(name)
        if not (d / "study.json").exists():
            raise KeyError(f"study '{name}' not found")
        return d

    def load(self, name: str) -> tuple[StudySpec, pd.DataFrame, pd.DataFrame]:
        d = self._dir(name)
        study = study_from_json(d / "study.json")
        res, mon = grid.load_results(d)
        return study, res, mon

    def summary(self, name: str, sweep: str) -> pd.DataFrame:
        _, res, _ = self.load(name)
        return grid.summarise(res, sweep)

    def curves(self, name: str, sweep: str) -> pd.DataFrame:
        _, res, mon = self.load(name)
        return grid.harvest_curves(mon, res, sweep)

    def concentrated(self, name: str, metric: str = "conc_months_to_diversify") -> pd.DataFrame:
        _, res, _ = self.load(name)
        return grid.concentrated_grid(res, metric)

    def report(self, name: str) -> str:
        study, res, mon = self.load(name)
        return markdown_report(study, res, mon)

    def export(self, name: str, path: str | Path | None = None) -> Path:
        study, res, mon = self.load(name)
        path = Path(path) if path else Path(self.ctx.settings.var_dir) / "exports" / f"tlh_research_{grid._slug(name)}.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        excel_report(study, res, mon, path)
        (path.with_suffix(".md")).write_text(markdown_report(study, res, mon), encoding="utf-8")
        return path

    def delete(self, name: str) -> None:
        import shutil
        shutil.rmtree(self._dir(name), ignore_errors=True)

    @staticmethod
    def approaches() -> dict:
        return dict(APPROACHES)

    @staticmethod
    def study_to_dict(study: StudySpec) -> dict:
        return asdict(study)
