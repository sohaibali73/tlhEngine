"""TLH Engine packaging.

Standard install (editable):          pip install -e .
One-click portable Windows EXE:       python setup.py build_exe        (or double-click Build-EXE.bat)

`build_exe` installs PyInstaller if missing and produces dist/TLHEngine/TLHEngine.exe (one-folder build, which is
the robust choice for QtWebEngine). Copy the whole dist/TLHEngine folder anywhere; on first run the app creates
`var/` and `.env` next to the executable. Norgate Data Updater must still be installed and running on that machine.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Command, find_packages, setup

ROOT = Path(__file__).resolve().parent
APP_NAME = "TLHEngine"


def _read_requirements() -> list[str]:
    req = ROOT / "requirements.txt"
    return [line.strip() for line in req.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")] if req.exists() else []


class BuildExe(Command):
    description = "build a portable Windows executable with PyInstaller (one-folder)"
    user_options = [("onefile", None, "single-file exe (slower start; not recommended with QtWebEngine)"),
                    ("clean", None, "remove build/ and dist/ first")]
    boolean_options = ["onefile", "clean"]

    def initialize_options(self):
        self.onefile = False
        self.clean = False

    def finalize_options(self):
        pass

    def run(self):
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print("PyInstaller not found; installing…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        if self.clean:
            for d in ("build", "dist"):
                shutil.rmtree(ROOT / d, ignore_errors=True)
        icon = _make_icon()
        sep = ";" if os.name == "nt" else ":"
        cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", APP_NAME, "--windowed",
               "--paths", str(ROOT),
               "--add-data", f"{ROOT / 'tlh' / 'db' / 'schema.sql'}{sep}tlh/db",
               "--add-data", f"{ROOT / 'tlh' / 'data' / 'substitutes.yaml'}{sep}tlh/data",
               "--add-data", f"{ROOT / '.env.example'}{sep}.",
               "--add-data", f"{ROOT / 'README.md'}{sep}.",
               "--add-data", f"{ROOT / 'DECISIONS.md'}{sep}.",
               "--collect-submodules", "tlh",
               "--collect-all", "cvxpy", "--collect-all", "clarabel", "--collect-all", "osqp", "--collect-all", "scs",
               "--collect-all", "plotly", "--collect-all", "duckdb", "--collect-all", "pyarrow", "--collect-all", "norgatedata",
               "--collect-all", "anthropic", "--collect-all", "sklearn", "--collect-all", "statsmodels", "--collect-all", "logbook",
               "--hidden-import", "PySide6.QtWebEngineWidgets", "--hidden-import", "PySide6.QtWebChannel", "--hidden-import", "PySide6.QtWebEngineCore",
               "--hidden-import", "scipy.special._cdflib", "--hidden-import", "pydantic_settings", "--hidden-import", "yaml",
               "--exclude-module", "tkinter", "--exclude-module", "matplotlib", "--exclude-module", "IPython", "--exclude-module", "pytest",
               ]
        if icon:
            cmd += ["--icon", str(icon)]
        cmd += ["--onefile"] if self.onefile else ["--onedir"]
        cmd.append(str(ROOT / "tlh_launcher.py"))
        print(" ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))
        out = ROOT / "dist" / APP_NAME
        if out.exists():
            shutil.copy2(ROOT / ".env.example", out / ".env.example")
            (out / "README-FIRST.txt").write_text(
                "TLH Engine portable build\n\n1. Norgate Data Updater must be installed and running.\n2. Start TLHEngine.exe. On first run it creates var/ and .env next to the exe.\n"
                "3. Enter your Anthropic API key in Settings > AI co-pilot (or edit .env) to enable YANG.\n4. Everything (database, snapshots, models, runs) lives in var/ next to the exe: copy the folder to move it.\n",
                encoding="utf-8")
        print(f"\nBuilt: {out / (APP_NAME + '.exe')}")


def _make_icon() -> Path | None:
    """Render the app icon to .ico with Pillow if available (optional)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    p = ROOT / "build" / "tlh.ico"
    p.parent.mkdir(exist_ok=True)
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, 248, 248), radius=56, fill=(59, 130, 246, 255))
    try:
        font = ImageFont.truetype("segoeuib.ttf", 150)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    d.text((128, 128), "T", fill="white", font=font, anchor="mm")
    img.save(p, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    return p


setup(
    name="tlh-engine",
    version="0.2.0",
    description="Tax-loss harvesting engine with factor risk model, optimizer, strategies and the YANG co-pilot",
    author="Sohaib Ali",
    python_requires=">=3.12",
    packages=find_packages(include=["tlh", "tlh.*"]),
    package_data={"tlh": ["db/schema.sql", "data/substitutes.yaml"]},
    include_package_data=True,
    install_requires=_read_requirements(),
    entry_points={"gui_scripts": ["tlh-engine=tlh.__main__:main"], "console_scripts": ["tlh=tlh.__main__:main"]},
    cmdclass={"build_exe": BuildExe},
)
