"""Application configuration.

Values come from (highest precedence first): explicit keyword overrides, the process environment, the repo-root
`.env` file, defaults. Secrets (ANTHROPIC_API_KEY, FRED_API_KEY) are read here and never persisted anywhere else.

This is a deliberately dependency-free loader (no pydantic-settings): it saves ~1.2 s of import time at launch and
the settings surface is small and flat.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

FROZEN = bool(getattr(sys, "frozen", False))
# Source checkout: repo root. Frozen (PyInstaller) build: the folder that holds the executable, so .env and var/ travel
# with the app. Bundled read-only resources (schema.sql, substitutes.yaml) live under sys._MEIPASS.
REPO_ROOT = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", REPO_ROOT))
ENV_FILE = REPO_ROOT / ".env"
if FROZEN and not ENV_FILE.exists():
    example = RESOURCE_ROOT / ".env.example"
    try:
        ENV_FILE.write_text(example.read_text(encoding="utf-8") if example.exists() else "ANTHROPIC_API_KEY=\n", encoding="utf-8")
    except OSError:
        pass

EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse KEY=value lines (comments, blank lines, `export` prefix and surrounding quotes tolerated)."""
    path = path or ENV_FILE
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            out[key.upper()] = val
    return out


def update_env_file(updates: dict[str, str], path: Path | None = None) -> Path:
    """Write KEY=value pairs into the .env file, preserving comments and unrelated lines. Creates the file if needed."""
    path = path or ENV_FILE
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped and not stripped.startswith("#") else None
        if key in updates:
            out.append(f"{key}={updates[key]}")
            done.add(key)
        else:
            out.append(line)
    for k, v in updates.items():
        if k not in done:
            out.append(f"{k}={v}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# attribute -> (ENV alias, default, caster)
_FIELDS: dict[str, tuple[str, Any, Any]] = {
    "anthropic_api_key": ("ANTHROPIC_API_KEY", "", str),
    "fred_api_key": ("FRED_API_KEY", "", str),
    "ai_model": ("TLH_AI_MODEL", "claude-opus-5", str),
    "ai_effort": ("TLH_AI_EFFORT", "high", str),
    "ai_max_tokens": ("TLH_AI_MAX_TOKENS", 16000, int),
    "var_dir": ("TLH_VAR_DIR", None, Path),          # None -> REPO_ROOT / var
    "default_universe": ("TLH_UNIVERSE", "S&P 500", str),
    "default_benchmark": ("TLH_BENCHMARK", "S&P 500", str),
    "price_history_start": ("TLH_PRICE_START", "2015-01-01", str),
    "ui_mode": ("TLH_UI_MODE", "simple", str),       # simple | expert (GUI default; can be changed in the app)
}
_ALIAS_TO_ATTR = {alias: attr for attr, (alias, _, _) in _FIELDS.items()}


class Settings:
    """Flat, immutable-by-convention settings object. Construct with `Settings()` or `Settings(TLH_VAR_DIR=...)`."""

    anthropic_api_key: str
    fred_api_key: str
    ai_model: str
    ai_effort: str
    ai_max_tokens: int
    var_dir: Path
    default_universe: str
    default_benchmark: str
    price_history_start: str
    ui_mode: str

    def __init__(self, _env_file: Path | None = None, **overrides: Any):
        file_vals = read_env_file(_env_file)
        env_vals = {k.upper(): v for k, v in os.environ.items() if k.upper() in _ALIAS_TO_ATTR}
        over = {}
        for k, v in overrides.items():
            alias = k.upper() if k.upper() in _ALIAS_TO_ATTR else _FIELDS.get(k, (None,))[0]
            if alias:
                over[alias] = v
        merged = {**file_vals, **env_vals, **over}
        for attr, (alias, default, caster) in _FIELDS.items():
            raw = merged.get(alias, default)
            if raw is None or (isinstance(raw, str) and raw.strip() == "" and caster is not str):
                raw = default
            if isinstance(raw, str):
                raw = raw.strip()
            try:
                val = caster(raw) if raw is not None else None
            except (TypeError, ValueError):
                val = default
            setattr(self, attr, val)
        if self.var_dir is None:
            self.var_dir = REPO_ROOT / "var"
        if self.ai_effort not in EFFORTS:
            self.ai_effort = "high"
        if self.ui_mode not in ("simple", "expert"):
            self.ui_mode = "simple"

    def __repr__(self) -> str:  # never echo the key
        return f"Settings(var_dir={self.var_dir}, ai_model={self.ai_model}, ai_effort={self.ai_effort}, key={'set' if self.anthropic_api_key else 'unset'})"

    def model_dump(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in _FIELDS}

    # Derived paths -------------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.var_dir / "tlh.sqlite"

    @property
    def cache_dir(self) -> Path:
        return self.var_dir / "cache"

    @property
    def snapshots_dir(self) -> Path:
        return self.cache_dir / "snapshots"

    @property
    def models_dir(self) -> Path:
        return self.var_dir / "models"

    @property
    def runs_dir(self) -> Path:
        return self.var_dir / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.var_dir / "exports"

    @property
    def sandbox_dir(self) -> Path:
        return self.var_dir / "sandbox"

    @property
    def logs_dir(self) -> Path:
        return self.var_dir / "logs"

    def ensure_dirs(self) -> None:
        for p in (self.var_dir, self.cache_dir, self.snapshots_dir, self.models_dir, self.runs_dir,
                  self.exports_dir, self.sandbox_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _settings
    if _settings is None or reload:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
