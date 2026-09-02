"""Live multi-turn co-pilot smoke (costs a few cents). Verifies history replay after tool use and streaming callbacks.
Run: python scripts/smoke_copilot.py"""
import logging
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

from tlh.ai.copilot import ChatCallbacks, Copilot  # noqa: E402
from tlh.services.context import AppContext  # noqa: E402

ctx = AppContext()
cp = Copilot(ctx)
cp.effort = "low"
cp.max_tokens = 4000
cid = cp.new_conversation("smoke: multi-turn")
events = {"text": 0, "thinking": 0, "tool_start": 0, "tool_end": 0}
cb = ChatCallbacks(
    on_text=lambda t: events.__setitem__("text", events["text"] + 1),
    on_thinking=lambda t: events.__setitem__("thinking", events["thinking"] + 1),
    on_tool_start=lambda i, n, a: (events.__setitem__("tool_start", events["tool_start"] + 1), print("  tool start:", n, a)),
    on_tool_end=lambda i, n, r, ok, s: (events.__setitem__("tool_end", events["tool_end"] + 1), print(f"  tool end: {n} ok={ok} {s:.1f}s")),
)
t = time.time()
r1 = cp.chat(cid, "Use get_portfolio_context and tell me in one sentence the total harvestable loss.", cb)
print(f"turn1 {r1.stop_reason} {time.time() - t:.0f}s ${r1.cost_usd:.3f}:", r1.text.strip()[:300])
t = time.time()
r2 = cp.chat(cid, "Now, in one sentence, which single lot is wash-blocked and why? (You already have the context.)", cb)
print(f"turn2 {r2.stop_reason} {time.time() - t:.0f}s ${r2.cost_usd:.3f}:", r2.text.strip()[:300])
print("events:", events, "| messages persisted:", len(ctx.conversations.messages(cid)))
assert r1.stop_reason == "end_turn" and r2.stop_reason == "end_turn", "turn failed"
print("MULTI-TURN OK")
