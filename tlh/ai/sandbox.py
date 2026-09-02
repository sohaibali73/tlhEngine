"""Local subprocess sandbox for AI-authored code.

A sandbox run copies the `tlh` package and `tests` into a fresh temp directory under var/sandbox, applies the
proposed file content, then runs pytest on the gating tests with a wall-clock timeout. The live database and
model artifacts are never touched: the sandbox gets its own TLH_VAR_DIR. Analysis scripts additionally receive
a *copy* of the SQLite state and read-only paths to the Parquet snapshots.

Why local, not Anthropic's server-side code execution: the code must run against Norgate data, the local
SQLite state and the live package, none of which exist in a remote container (DECISIONS.md D6).
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import REPO_ROOT, get_settings
from .registry import is_editable, tests_for

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".hypothesis")


@dataclass
class SandboxResult:
    passed: bool
    returncode: int
    stdout: str
    duration_s: float
    workdir: str
    tests: list[str] = field(default_factory=list)
    syntax_error: str | None = None

    def summary(self, max_chars: int = 6000) -> str:
        head = ("PASSED" if self.passed else "FAILED") + f" (rc={self.returncode}, {self.duration_s:.1f}s)"
        if self.syntax_error:
            return f"{head}\nSyntax error: {self.syntax_error}"
        out = self.stdout
        if len(out) > max_chars:
            out = out[:max_chars // 2] + "\n...[truncated]...\n" + out[-max_chars // 2:]
        return f"{head}\nTests: {', '.join(self.tests)}\n{out}"


def _new_workdir(prefix: str) -> Path:
    root = get_settings().sandbox_dir
    root.mkdir(parents=True, exist_ok=True)
    wd = root / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    wd.mkdir()
    return wd


def _stage_repo(wd: Path) -> None:
    if not (REPO_ROOT / "tlh").exists() or not (REPO_ROOT / "tests").exists():
        raise RuntimeError("The code sandbox needs the source checkout (tlh/ and tests/ next to the app). "
                           "It is unavailable in the portable EXE build; run from source to let YANG propose code changes.")
    shutil.copytree(REPO_ROOT / "tlh", wd / "tlh", ignore=IGNORE)
    shutil.copytree(REPO_ROOT / "tests", wd / "tests", ignore=IGNORE)
    for name in ("pyproject.toml",):
        if (REPO_ROOT / name).exists():
            shutil.copy2(REPO_ROOT / name, wd / name)
    (wd / "var").mkdir(exist_ok=True)


def _env(wd: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["TLH_VAR_DIR"] = str(wd / "var")
    env["PYTHONPATH"] = str(wd)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TLH_SANDBOX"] = "1"
    env.pop("ANTHROPIC_API_KEY", None)        # sandboxed code never gets the key
    return env


def check_syntax(path: str, code: str) -> str | None:
    if path.endswith(".py"):
        try:
            ast.parse(code, filename=path)
        except SyntaxError as e:
            return f"{e.msg} (line {e.lineno})"
    elif path.endswith((".yaml", ".yml")):
        import yaml
        try:
            yaml.safe_load(code)
        except yaml.YAMLError as e:
            return str(e)
    return None


def run_tests_with_change(module_path: str, new_code: str, tests: list[str] | None = None,
                          timeout_s: int = 600, extra_files: dict[str, str] | None = None) -> SandboxResult:
    """Apply `new_code` at `module_path` inside a staged copy and run the gating tests."""
    module_path = module_path.replace("\\", "/")
    if not is_editable(module_path):
        raise PermissionError(f"{module_path} is not an AI-editable module")
    tests = tests or tests_for(module_path)
    err = check_syntax(module_path, new_code)
    wd = _new_workdir("test")
    if err:
        return SandboxResult(False, -1, "", 0.0, str(wd), tests, syntax_error=err)
    _stage_repo(wd)
    target = wd / module_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new_code, encoding="utf-8")
    for p, txt in (extra_files or {}).items():
        fp = wd / p.replace("\\", "/")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(txt, encoding="utf-8")
    cmd = [sys.executable, "-m", "pytest", *tests, "-q", "-x", "-p", "no:cacheprovider", "--no-header"]
    return _run(cmd, wd, timeout_s, tests)


def run_analysis(code: str, timeout_s: int = 300, copy_db: bool = True) -> SandboxResult:
    """Run an ad-hoc analysis script against a COPY of the state DB and the real (read-only) snapshot files."""
    wd = _new_workdir("analysis")
    _stage_repo(wd)
    settings = get_settings()
    if copy_db and settings.db_path.exists():
        shutil.copy2(settings.db_path, wd / "var" / "tlh.sqlite")
    # expose snapshot/model folders read-only by path (the script is told where they live)
    preamble = (
        "import os, sys\n"
        f"SNAPSHOTS_DIR = r'{settings.snapshots_dir}'\n"
        f"MODELS_DIR = r'{settings.models_dir}'\n"
        f"RUNS_DIR = r'{settings.runs_dir}'\n"
        "os.environ.setdefault('TLH_VAR_DIR', os.path.join(os.getcwd(), 'var'))\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
    )
    script = wd / "analysis.py"
    script.write_text(preamble + code, encoding="utf-8")
    return _run([sys.executable, str(script)], wd, timeout_s, [])


def _run(cmd: list[str], wd: Path, timeout_s: int, tests: list[str]) -> SandboxResult:
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(wd), env=_env(wd), capture_output=True, text=True, timeout=timeout_s,
                              encoding="utf-8", errors="replace")
        out = (proc.stdout or "") + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out += f"\n[timeout after {timeout_s}s]"
        rc = -9
    dur = time.time() - t0
    return SandboxResult(passed=(rc == 0), returncode=rc, stdout=out, duration_s=dur, workdir=str(wd), tests=tests)


def cleanup(keep_last: int = 20) -> None:
    root = get_settings().sandbox_dir
    if not root.exists():
        return
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    for p in dirs[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)
