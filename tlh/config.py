"""Application configuration.

Values come from (highest precedence first): process environment, the repo-root `.env` file, defaults.
Secrets (ANTHROPIC_API_KEY, FRED_API_KEY) are read here and never persisted anywhere else.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Secrets
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")

    # AI
    ai_model: str = Field(default="claude-opus-5", alias="TLH_AI_MODEL")
    ai_effort: str = Field(default="high", alias="TLH_AI_EFFORT")
    ai_max_tokens: int = Field(default=16000, alias="TLH_AI_MAX_TOKENS")

    # Paths
    var_dir: Path = Field(default=REPO_ROOT / "var", alias="TLH_VAR_DIR")

    # Data defaults
    default_universe: str = Field(default="S&P 500", alias="TLH_UNIVERSE")
    default_benchmark: str = Field(default="S&P 500", alias="TLH_BENCHMARK")
    price_history_start: str = Field(default="2015-01-01", alias="TLH_PRICE_START")

    @field_validator("anthropic_api_key", "fred_api_key", "ai_model", "ai_effort", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("ai_effort")
    @classmethod
    def _effort(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "xhigh", "max"}
        return v if v in allowed else "high"

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
