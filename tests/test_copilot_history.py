from tlh.ai.copilot import _repair_history, estimate_cost, sanitize_block, sanitize_content


def test_sanitize_drops_sdk_only_and_null_fields():
    b = {"type": "text", "text": "hi", "citations": None, "parsed_output": None}
    assert sanitize_block(b) == {"type": "text", "text": "hi"}
    tu = {"type": "tool_use", "id": "t1", "name": "x", "input": {"a": 1}, "caller": {"type": "direct"}}
    assert sanitize_block(tu) == {"type": "tool_use", "id": "t1", "name": "x", "input": {"a": 1}}
    th = {"type": "thinking", "thinking": "", "signature": "sig"}
    assert sanitize_block(th) == th                      # empty string is not None: keep
    assert sanitize_content("plain") == "plain"


def test_repair_history_drops_dangling_tool_use():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "t", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "t", "input": {}}]},   # cancelled: no result
    ]
    out = _repair_history(msgs)
    assert len(out) == 5
    assert out[-1] == {"role": "user", "content": "q2"}


def test_repair_history_strips_leading_non_user():
    msgs = [{"role": "assistant", "content": [{"type": "text", "text": "x"}]}, {"role": "user", "content": "q"}]
    assert _repair_history(msgs)[0]["role"] == "user"


def test_estimate_cost():
    u = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert estimate_cost("claude-opus-5", u) == 5.0 + 2.5
    assert estimate_cost("unknown-model", u) == 7.5
