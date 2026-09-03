# Getting started after cloning

This gets a fresh clone of the TLH Engine running on a Windows machine, from nothing to the GUI with live data, the
test suite green, and YANG (the embedded co-pilot) answering. Budget about 30 minutes, most of it downloads.

## 0. What you need before you start

| Requirement | Why | Where |
|---|---|---|
| Windows 10/11, 64-bit | PySide6 + QtWebEngine and Norgate's updater are Windows builds | |
| Python 3.12 (64-bit) | the code uses 3.12 syntax; 3.13 is untested | https://www.python.org/downloads/ (tick "Add to PATH") |
| Norgate Data Updater (NDU) with a US Equities subscription | every price, fundamental and watchlist comes from Norgate; there is no other data path | https://norgatedata.com/ndu.php |
| Git | to clone and push | https://git-scm.com/ |
| An Anthropic API key | only for YANG (chat, pop-up, agent). Everything else runs without it | https://console.anthropic.com/ |
| Optional: a FRED API key | macro series fallback; Norgate's Economic database is used by default | https://fred.stlouisfed.org/docs/api/api_key.html |

Install NDU first, sign in, and let it finish its initial download. The app polls NDU every 10 seconds and shows a
banner until it is running.

## 1. Clone and create a virtual environment

Open PowerShell:

```powershell
git clone https://github.com/sohaibali73/tlhEngine.git
cd tlhEngine
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

## 2. Install the package and its dependencies

```powershell
pip install -e ".[dev]"
```

This pulls PySide6, cvxpy with the CLARABEL/OSQP/SCS solvers, DuckDB, pyarrow, plotly, the Anthropic SDK, pytest and
ruff. The first install is about 1 GB. `-e` means edits to `tlh/` take effect without reinstalling.

Check the two things that most often go wrong:

```powershell
python -c "import norgatedata, PySide6, cvxpy; print('imports ok')"
python -c "import norgatedata; print(norgatedata.status())"
```

The second line must print `True`. If it prints `False`, NDU is not running or not signed in.

## 3. Create your `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in `ANTHROPIC_API_KEY=` and save. Everything else can stay at its default. `.env` is git-ignored, so the key
never leaves your machine. You can also enter the key later inside the app under Settings, which writes the same file.

Do not put the key anywhere else: not in code, not in the database, not in an export.

## 4. First launch

```powershell
python -m tlh --seed-demo
```

The `--seed-demo` flag creates a demo household with accounts and tax lots so every screen has something to show. It is
idempotent, so running it again does nothing harmful. What happens on first launch:

1. A splash appears, then the window lands on **Start here** in about two seconds.
2. In the background the app pulls an S&P 500 snapshot from Norgate (about 640 symbols including the leveraged and
   inverse ETFs, ten years of history, 20 to 60 seconds) and writes it as Parquet under `var/`.
3. It then fits the default risk model. The status bar shows progress. Until the fit finishes, screens that need a
   model show a hint instead of numbers.

When the status bar is quiet, click **Find my tax savings** on Start here. That runs the whole pipeline once.

Normal launches afterwards:

```powershell
python -m tlh            # simple mode: Start here, Portfolio, Harvest, Risk model, Concentration, Model portfolios, Tactical overlay, Tax rates, YANG, Settings
python -m tlh --expert   # every quant workbench (Risk lab, Strategy lab, TLH model builder, YANG Agent, Export)
```

Press **F1** inside the app for the guided tour, or **Ctrl+H** to return to Start here.

## 5. Prove the install is healthy

Run these in order. Each one should end the way shown.

```powershell
python -m pytest                                   # "231 passed"; the wash-sale and lot tests are the compliance gate
ruff check tlh tests                               # "All checks passed!"
$env:PYTHONPATH = "."; python scripts/smoke_tools.py    # ends "63 tools defined; failures: none"
$env:QT_QPA_PLATFORM = "offscreen"; python scripts/smoke_gui.py   # ends "GUI errors: none"
Remove-Item Env:QT_QPA_PLATFORM
python scripts/smoke_copilot.py                    # live two-turn YANG exchange; costs a few cents, needs the API key
```

The tool and GUI smokes run against your real `var/` state, so run them after the first launch has pulled data and
fitted a model.

## 6. Where things live

| Path | What it is |
|---|---|
| `tlh/` | the application. Layers are listed in CLAUDE.md; architecture in docs/ARCHITECTURE.md |
| `var/` | runtime state: SQLite database, Parquet snapshots, fitted models, run artifacts, exports. Git-ignored. Delete it to start clean |
| `.env` | secrets and overrides. Git-ignored |
| `DECISIONS.md` | every stack and tax-rule decision with its reasoning. Read D1 to D21 before changing tax, optimizer or AI-persistence code |
| `docs/WHITEPAPER.md` | the algorithms and the operating model in prose |
| `TLH_Engine_with_YANG___Technical___Positioning_Write-Up/` | the Potomac-branded print version of the whitepaper. Check layout with `python scripts/render_writeup.py` |
| `Tax Loss Harvesting/` | source research documents. Git-ignored and not needed to run |

To keep `var/` somewhere else, set `TLH_VAR_DIR` in `.env`.

## 7. Working on the code

- Run `python -m pytest` before every push. Tax and wash-sale changes need a hand-computed expected value in
  `tests/test_washsale.py` or `tests/test_holding.py`.
- `ruff check tlh tests` must be clean.
- Adding a strategy, a risk-model preset, a sample basket or a YANG tool each has a one-line recipe in CLAUDE.md.
- GUI code never computes and services never import Qt. Heavy imports go through `tlh/lazy.py`.
- Nothing in this program places orders. Trade tickets are the terminal output. Keep it that way.

## 8. Building the standalone EXE (optional)

```powershell
pip install pyinstaller
.\Build-EXE.bat
```

About 13 minutes. The result in `dist/` launches without Python installed and creates its own `var/` and `.env` beside
the executable. Norgate Data Updater still has to be installed and running on that machine.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Banner says Norgate Data Updater is not running | Start NDU and sign in. The banner clears within 10 seconds |
| `python -m tlh` opens then closes | Run it from a terminal to see the traceback. Usually a missing package: rerun step 2 |
| YANG says it is unavailable | `ANTHROPIC_API_KEY` is empty or wrong in `.env`. Fix it in Settings, which reconfigures YANG immediately |
| Blank charts | QtWebEngine needs a GPU or software OpenGL. Update graphics drivers; on a VM set `QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu` |
| Console shows garbled characters | Set `PYTHONIOENCODING=utf-8`; the Windows console defaults to cp1252 |
| Snapshot pull fails with a UNIQUE error | Two pulls started at once. Wait for the first to finish; the GUI now guards against this |
| Tests fail only in `test_leverage.py` | The leveraged ETFs are missing from your snapshot. Refresh data from the Portfolio screen so the universe includes them |
