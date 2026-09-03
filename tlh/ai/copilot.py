"""Embedded Claude co-pilot: streamed chat + tool loop + the draft -> sandbox -> approve -> promote lifecycle.

Runtime subsystem (distinct from Claude Code, which built the app). Manual tool loop over the Anthropic Messages API
so the GUI can stream text and reasoning, show live tool calls, cancel, gate code changes behind human approval and
persist every turn. See DECISIONS.md D6.
"""
from __future__ import annotations

import difflib
import importlib
import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, get_settings
from ..services.context import AppContext
from . import sandbox
from .registry import is_editable
from .tools import TOOLS, ToolExecutor

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are YANG, the embedded quantitative co-pilot (built on Claude) inside a tax-loss-harvesting (TLH) engine \
used by a director of quant research. Refer to yourself as YANG. You design and revise the equity factor risk model, construct model portfolios (baskets), \
design harvest trade plans, run and tune the harvest optimizer, maintain the substantially-identical-security mapping, \
and explain wash-sale determinations and recommendations precisely.

Capabilities (all via tools)
- Ground everything in live state: portfolio, lots, wash calendar, active risk model, past runs, price statistics.
- Model portfolios: `optimize_basket` (min-TE construction with name caps, sector band, style tilts, exclusions) or \
`create_basket` from explicit weights; `analyze_basket`; `set_benchmark` to 'basket:<name>' to harvest toward a model.
- Strategies: `list_strategies`, then `build_strategy_basket` for minimum variance, maximum diversification, risk parity, \
HRP, mean-variance with signal alphas, Black-Litterman with views, minimum CVaR, stratified direct-index sampling, factor \
tilts, and tax-aware transition (move the live portfolio toward a target under a realised-gain budget). \
`backtest_strategy` walks any of them forward with costs and reports CAGR/Sharpe/drawdown/TE/IR — always relay its \
caveats (survivorship, fundamental look-ahead) when quoting results. New strategies can be added to tlh/optim/strategies.py \
via the code-change loop.
- TLH model pipelines (the drag-and-drop builder): `pipeline_schema` lists the blocks; `save_pipeline` stores a pipeline you design as \
JSON (Universe -> Filter/Rank -> Benchmark -> Construction -> Transition -> Harvest -> Output); `run_pipeline` executes a saved or inline \
pipeline end to end and returns the basket, harvest run and log. `list_pipelines` shows saved models.
- Concentrated positions & embedded gains: `concentration_overview` (weights, embedded gains, tax if sold, risk share, HHI), \
`diversification_plan` (multi-year bracket-aware glide path with loss offsets, step-up, alpha view), `concentration_monte_carlo`, \
`hedge_analysis` (collars with constructive-sale/straddle flags), `concentration_alternatives` (donate vs sell, gifting, exchange fund, \
step-up), and `gain_offset_plan` (pair a concentrated sale with harvestable losses as a reviewable trade plan).
- Harvesting: `run_harvest` with config overrides (mode, priority order, TE budget, drift limits, target loss, benchmark); \
`run_frontier`; or design your own plan and `evaluate_trade_list` — every trade is wash-screened and TE before/after is \
computed; the user reviews it in the Harvest screen.
- Advisor onboarding: `import_holdings` (broker CSV/Excel -> lots), `set_tax_setup` (state + filing + income -> marginal rates), \
`state_tax_rates` (every state's capital-gains treatment, combined federal+NIIT+state), `one_click_harvest` (the Start-here flow with a \
plain-English summary). Talk to advisors in plain English; quote the tax value of a loss in dollars.
- Sample model portfolios: `build_sample_baskets` builds the 17-recipe library (index trackers, integrated multi-factor, defensive \
equity, quality-momentum, risk parity, HRP, style tilts, min-CVaR, Black-Litterman, 130/30 and 145/45 long/short tax engines). New strategies: \
`multi_factor` (integrated vs mixed), `defensive_equity` (beta cap), `quality_momentum`, `long_short_extension` (130/30-style, beta ~1, \
sector/style-neutral extension), `overlay_neutral` (market-neutral extension around existing holdings, never shorts held names).
- Long/short TLH economics: `longshort_analysis` (loss generation by year vs long-only, financing, net tax benefit, Quantinno reference), \
`exchange_glide` (tax-neutral divestiture of a concentrated stock funded by extension losses), `overlay_plan` (index-futures beta overlay: \
contracts, margin, carry, §1256 60/40 and straddle flags). Shorting and futures require custodian capability and tax counsel; say so.
- Risk models: a library of 13 presets (`risk_model_presets`; fit with `fit_risk_model(preset=...)`): ERM standard / short horizon / long horizon / \
robust / GARCH-dynamic / regime-conditional, hybrid ERM + statistical residual factors, Potomac calibrated covariances (126d equal Ledoit-Wolf; \
189d exponential), tight-pair sample covariance, PCA, barra_lite. `run_calibration_study` (lookback x weighting x estimator x horizon walk-forward, \
port of the 2026 Potomac study) and `pair_te_study` tell you which estimator to trust for baskets versus tight substitute pairs.
- Estimators — `barra_lite` and the full equity risk model `erm` (multi-descriptor styles incl. resvol/liquidity/\
leverage/midcap, GICS industry groups, Huber option, Newey-West, eigen-adjusted and regime-adjusted covariance, shrunk specific risk). \
`fit_risk_model` with overrides such as {model_kind:'erm', industry_level:'gics_industry_group', robust:true} creates a version; \
`risk_decomposition`, `stress_test` (sigma or raw factor shocks, propagated), `historical_scenario`, `parametric_var` and \
`validate_risk_model` (out-of-sample bias statistics) analyse it. `fit_risk_model` with spec overrides (styles, lookback, half-life, macro) creates a new version to compare \
with `compare_models`; invent new style factors by proposing a module under tlh/risk/custom/ that registers into \
STYLE_DEFINITIONS, then fit with that style included. Edit factors.py / model.py / harvest.py / basket.py / \
substitutes.yaml via `read_module` -> `test_change` (iterate) -> `propose_change` (once, when tests pass).

Ground rules
- You never place trades. Nothing you do bypasses the wash-sale engine; the tools screen for you.
- Code changes never touch the live model directly: `propose_change` runs the sandbox and the human approves in the UI. \
Keep public interfaces stable (FittedRiskModel, HarvestConfig/HarvestInputs/HarvestResult, STYLE_DEFINITIONS, BasketSpec).
- Tax rules (holding period, 61-day wash window, netting) are fixed conventions in DECISIONS.md D8; do not reinterpret \
them. If asked for tax advice, explain the engine's logic and defer to a tax advisor.
- Be concise, quantitative, and direct: this is a PM-desk tool. Quote numbers with units. Use markdown tables for \
trade lists and comparisons. When you finish a multi-step task, summarise what was created (run ids, basket names, \
change ids) so the user can find it in the GUI.
"""

# $ per million tokens: (input, output, cache_write, cache_read)
PRICES = {
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-sonnet-5": (2.0, 10.0, 2.50, 0.20),
    "claude-fable-5-1": (10.0, 50.0, 12.50, 1.00),
    "claude-opus-4-8": (5.0, 25.0, 6.25, 0.50),
}

DROP_KEYS = {"parsed_output", "caller"}


def sanitize_block(b: dict) -> dict:
    """Strip SDK-only / null fields so persisted content can be replayed to the API."""
    return {k: v for k, v in b.items() if k not in DROP_KEYS and v is not None}


def sanitize_content(content: Any) -> Any:
    if isinstance(content, list):
        return [sanitize_block(b) if isinstance(b, dict) else b for b in content]
    return content


def estimate_cost(model: str, usage: dict) -> float:
    p = PRICES.get(model, PRICES["claude-opus-5"])
    return (usage.get("input_tokens", 0) * p[0] + usage.get("output_tokens", 0) * p[1]
            + usage.get("cache_creation_input_tokens", 0) * p[2] + usage.get("cache_read_input_tokens", 0) * p[3]) / 1e6


@dataclass
class TurnResult:
    text: str
    stop_reason: str
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    change_ids: list[int] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class ChatCallbacks:
    on_text: Callable[[str], None] | None = None
    on_thinking: Callable[[str], None] | None = None
    on_tool_start: Callable[[str, str, dict], None] | None = None          # (tool_use_id, name, args)
    on_tool_end: Callable[[str, str, str, bool, float], None] | None = None  # (tool_use_id, name, result_preview, ok, seconds)
    on_status: Callable[[str], None] | None = None


class Copilot:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.settings = get_settings()
        self._client = None
        self.model = self.settings.ai_model
        self.effort = self.settings.ai_effort
        self.max_tokens = self.settings.ai_max_tokens
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ client
    @property
    def available(self) -> bool:
        return bool(self.settings.anthropic_api_key)

    @property
    def client(self):
        if self._client is None:
            import anthropic
            if not self.settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (add it to .env and restart).")
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def cancel(self) -> None:
        self._cancel.set()

    def reconfigure(self) -> None:
        """Re-read .env (API key, model, effort) and drop the cached client. Called after Settings saves a key."""
        self.settings = get_settings(reload=True)
        self._client = None
        self.model = self.settings.ai_model
        self.effort = self.settings.ai_effort
        self.max_tokens = self.settings.ai_max_tokens

    # ------------------------------------------------------------------ conversations
    def new_conversation(self, title: str | None = None) -> int:
        return self.ctx.conversations.create(self.model, title)

    def history(self, conversation_id: int) -> list[dict]:
        out = []
        for m in self.ctx.conversations.messages(conversation_id):
            c = sanitize_content(m["content"])
            if m["role"] == "assistant" and isinstance(c, list) and not c:
                continue
            out.append({"role": m["role"], "content": c})
        return _repair_history(out)

    def chat(self, conversation_id: int, user_text: str, cb: ChatCallbacks | None = None, max_iterations: int = 40,
             extra_system: str | None = None) -> TurnResult:
        """One user turn including tool round-trips. Streams text/thinking/tool events through `cb`."""
        cb = cb or ChatCallbacks()
        self._cancel.clear()
        t0 = time.time()
        messages = self.history(conversation_id)
        messages.append({"role": "user", "content": user_text})
        self.ctx.conversations.add_message(conversation_id, "user", user_text)
        executor = ToolExecutor(self.ctx, conversation_id)
        out_text: list[str] = []
        change_ids: list[int] = []
        tool_log: list[dict] = []
        usage_total: dict[str, int] = {}
        stop = "end_turn"
        for _ in range(max_iterations):
            if cb.on_status:
                cb.on_status("thinking")
            partial: list[str] = []
            with self.client.messages.stream(
                model=self.model, max_tokens=self.max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
                + ([{"type": "text", "text": extra_system}] if extra_system else []),
                messages=messages, tools=TOOLS,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": self.effort},
            ) as stream:
                for event in stream:
                    if self._cancel.is_set():
                        break
                    et = getattr(event, "type", "")
                    if et == "text":
                        partial.append(event.text)
                        if cb.on_text:
                            cb.on_text(event.text)
                    elif et == "thinking" and cb.on_thinking:
                        cb.on_thinking(getattr(event, "thinking", "") or "")
                    elif et == "content_block_start" and cb.on_status:
                        blk = getattr(event, "content_block", None)
                        if blk is not None and getattr(blk, "type", "") == "tool_use":
                            cb.on_status(f"calling {blk.name}")
                if self._cancel.is_set():
                    text = "".join(partial)
                    out_text.append(text)
                    if text.strip():
                        self.ctx.conversations.add_message(conversation_id, "assistant", [{"type": "text", "text": text + "\n\n[stopped by user]"}])
                    stop = "cancelled"
                    break
                msg = stream.get_final_message()
            out_text.append("".join(partial))
            content_dump = [sanitize_block(b.model_dump()) for b in msg.content]
            u = msg.usage.model_dump() if msg.usage else {}
            for k, v in u.items():
                if isinstance(v, int):
                    usage_total[k] = usage_total.get(k, 0) + v
            self.ctx.conversations.add_message(conversation_id, "assistant", content_dump, usage=u)
            messages.append({"role": "assistant", "content": msg.content})
            stop = msg.stop_reason or "end_turn"
            if stop == "refusal":
                note = "\n[The model declined this request (safety refusal)."
                det = getattr(msg, "stop_details", None)
                if det is not None and getattr(det, "category", None):
                    note += f" Category: {det.category}."
                note += "]"
                out_text.append(note)
                if cb.on_text:
                    cb.on_text(note)
                break
            if stop != "tool_use":
                if stop == "max_tokens":
                    note = "\n[Response truncated at max_tokens.]"
                    out_text.append(note)
                    if cb.on_text:
                        cb.on_text(note)
                break
            results = []
            for block in msg.content:
                if block.type != "tool_use":
                    continue
                args = block.input if isinstance(block.input, dict) else {}
                if cb.on_tool_start:
                    cb.on_tool_start(block.id, block.name, _trim_args(args))
                ts = time.time()
                try:
                    result, cid = executor.execute(block.name, args)
                    is_err = False
                except Exception as e:  # tool errors go back to the model, never crash the loop
                    log.exception("tool %s failed", block.name)
                    result, cid, is_err = f"Error: {type(e).__name__}: {e}", None, True
                dt = time.time() - ts
                if cid:
                    change_ids.append(cid)
                tool_log.append({"name": block.name, "input": _trim_args(args), "ok": not is_err, "seconds": round(dt, 2)})
                if cb.on_tool_end:
                    cb.on_tool_end(block.id, block.name, result[:600], not is_err, dt)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": is_err})
                if self._cancel.is_set():
                    break
            messages.append({"role": "user", "content": results})
            self.ctx.conversations.add_message(conversation_id, "user", results)
            if self._cancel.is_set():
                stop = "cancelled"
                break
        return TurnResult(text="".join(out_text), stop_reason=stop, usage=usage_total, cost_usd=estimate_cost(self.model, usage_total),
                          change_ids=change_ids, tool_calls=tool_log, duration_s=time.time() - t0)

    # ------------------------------------------------------------------ change lifecycle
    def propose(self, path: str, code: str, title: str, rationale: str, conversation_id: int | None) -> int:
        return propose(self.ctx, path, code, title, rationale, conversation_id)

    def retest(self, change_id: int) -> bool:
        ch = self.ctx.code.change(change_id)
        res = sandbox.run_tests_with_change(ch["module_path"], ch["proposed_code"])
        self.ctx.code.set_sandbox_result(change_id, res.summary(max_chars=20000), res.passed)
        return res.passed

    def approve_and_promote(self, change_id: int, approved_by: str = "user", force: bool = False) -> int:
        ch = self.ctx.code.change(change_id)
        if ch is None:
            raise KeyError(change_id)
        if ch["status"] in ("promoted", "rejected"):
            raise RuntimeError(f"change #{change_id} is already {ch['status']}")
        if not ch["sandbox_passed"] and not force:
            if not self.retest(change_id):
                raise RuntimeError("sandbox tests fail; approve with force=True to promote anyway (not recommended)")
        path = ch["module_path"]
        fp = REPO_ROOT / path
        if self.ctx.code.latest_version(path) is None and fp.exists():
            self.ctx.code.add_version(path, fp.read_text(encoding="utf-8"), "human")
        self.ctx.code.set_status(change_id, "approved", approved_by=approved_by)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(ch["proposed_code"], encoding="utf-8")
        vid = self.ctx.code.add_version(path, ch["proposed_code"], "ai", change_id=change_id)
        self.ctx.code.set_status(change_id, "promoted", promoted_version_id=vid)
        self.ctx.db.audit("user", "change.promote", path, change_id=change_id, version_id=vid, approved_by=approved_by)
        self.hot_reload(path)
        return vid

    def reject(self, change_id: int, reason: str = "") -> None:
        self.ctx.code.set_status(change_id, "rejected", approved_by="user")
        self.ctx.db.audit("user", "change.reject", str(change_id), reason=reason)

    def rollback(self, module_path: str, version_id: int) -> int:
        v = self.ctx.code.get_version(version_id)
        if v is None or v["module_path"] != module_path:
            raise KeyError(version_id)
        fp = REPO_ROOT / module_path
        fp.write_text(v["code_text"], encoding="utf-8")
        new_vid = self.ctx.code.add_version(module_path, v["code_text"], "rollback")
        self.ctx.db.audit("user", "code.rollback", module_path, to_version=version_id, new_version=new_vid)
        self.hot_reload(module_path)
        return new_vid

    RELOAD_ORDER = ["tlh.risk.factors", "tlh.risk.model", "tlh.optim.harvest", "tlh.optim.frontier", "tlh.optim.basket",
                    "tlh.risk.descriptors", "tlh.risk.erm", "tlh.risk.analytics",
                    "tlh.services.risk_service", "tlh.services.harvest_service", "tlh.services.basket_service"]

    def hot_reload(self, module_path: str) -> list[str]:
        """Reload the changed module and everything downstream of it (in dependency order)."""
        if module_path.endswith((".yaml", ".yml")):
            self.ctx.reload_substitutes()
            return ["substitutes"]
        modname = module_path[:-3].replace("/", ".")
        if modname.startswith("tlh.risk.custom."):
            from ..risk.custom import load_all
            return load_all(reload=True)
        order = list(self.RELOAD_ORDER)
        if modname not in order:
            order.insert(0, modname)
        start = order.index(modname)
        reloaded = []
        for m in order[start:]:
            if m in sys.modules:
                importlib.reload(sys.modules[m])
                reloaded.append(m)
        if modname == "tlh.risk.factors":
            from ..risk.custom import load_all
            reloaded += load_all(reload=True)      # custom plugins register into the fresh STYLE_DEFINITIONS
        return reloaded


def propose(ctx: AppContext, path: str, code: str, title: str, rationale: str, conversation_id: int | None) -> int:
    path = path.replace("\\", "/")
    if not is_editable(path):
        raise PermissionError(f"{path} is not AI-editable (see tlh/ai/registry.py)")
    current = (REPO_ROOT / path).read_text(encoding="utf-8") if (REPO_ROOT / path).exists() else ""
    diff = "".join(difflib.unified_diff(current.splitlines(True), code.splitlines(True), fromfile=f"a/{path}", tofile=f"b/{path}"))
    cid = ctx.code.create_change(path, title, code, rationale, diff, conversation_id)
    res = sandbox.run_tests_with_change(path, code)
    ctx.code.set_sandbox_result(cid, res.summary(max_chars=20000), res.passed)
    return cid


def _repair_history(msgs: list[dict]) -> list[dict]:
    """Guarantee a replayable sequence: every tool_use must be followed by matching tool_results, and the list must
    alternate sensibly. Drops dangling tool_use turns (e.g. from a cancelled run)."""
    out: list[dict] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant" and isinstance(m["content"], list) and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in m["content"]):
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            ids = {b["id"] for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"}
            have = {b.get("tool_use_id") for b in (nxt["content"] if nxt and isinstance(nxt["content"], list) else [])
                    if isinstance(b, dict) and b.get("type") == "tool_result"}
            if not nxt or nxt["role"] != "user" or not ids <= have:
                i += 1                 # dangling tool_use: drop it
                continue
        out.append(m)
        i += 1
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def _trim_args(args: dict, n: int = 200) -> dict:
    return {k: (v if not isinstance(v, str) or len(v) <= n else v[:n] + f"... [{len(v)} chars]") for k, v in args.items()}


def sandbox_root() -> Path:
    return get_settings().sandbox_dir
