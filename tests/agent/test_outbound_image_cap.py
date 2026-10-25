"""A declared per-request image limit (``max_images``) reaches the production send path.

A vLLM server started with ``--limit-mm-per-prompt '{"image": 8}'`` refuses a ninth image with
``At most 8 image(s) may be provided in one prompt.`` (HTTP 400, not retried). Hermes' own
screenshots are within the many-image cap, so the default ceiling is 100 and a long screenshot
turn sends every one; with ``model.max_images: 8`` request assembly retires the old ones first.
"""

from __future__ import annotations

import base64
import copy
import logging
from types import SimpleNamespace

import pytest

ROUTE = ("custom", "http://localhost:8000/v1")
MODEL = "local-vision-model"


def _png(width: int = 1568, height: int = 1000) -> str:
    """Header-only PNG data URL: all the eviction policy reads to classify an image."""
    ihdr = b"IHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"
    raw = b"\x89PNG\r\n\x1a\n" + len(ihdr).to_bytes(4, "big") + ihdr
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _screenshot_turn(n: int) -> list[dict]:
    """One user ask, then ``n`` browser steps that each returned a screenshot (a mid-turn call)."""
    history: list[dict] = [{"role": "user", "content": "check the page"}]
    for i in range(n):
        history.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "browser_exec", "arguments": "{}"}}],
        })
        history.append({
            "role": "tool", "tool_call_id": f"call_{i}",
            "content": [{"type": "text", "text": f"step {i}"}, {"type": "image_url", "image_url": {"url": _png()}}],
        })
    return history


def _image_parts(messages: list[dict]) -> int:
    return sum(
        1
        for msg in messages
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )


def _agent(tmp_path):
    from agent.agent_runtime_helpers import copy_reasoning_content_for_api
    from agent.context_compressor import ContextCompressor
    from agent.message_sanitization import ReasoningReplayRoute
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli", model=MODEL)
    agent = SimpleNamespace(
        api_mode="chat_completions", provider=ROUTE[0], requested_provider=ROUTE[0], model=MODEL,
        base_url=ROUTE[1], session_id="s1", _session_db=db, _persist_disabled=False,
        _current_turn_timestamp=1.0, ephemeral_system_prompt="", prefill_messages=[],
        tools=[], _use_prompt_caching=False, _usage_anchor=None, verbose_logging=False,
        reasoning_callback=None, stream_delta_callback=None, _stream_callback=None, log_prefix="",
    )
    agent._needs_thinking_reasoning_pad = lambda: False
    agent._reasoning_replay_route = lambda: ReasoningReplayRoute(False, None)
    agent._copy_reasoning_content_for_api = (
        lambda source, target: copy_reasoning_content_for_api(agent, source, target)
    )
    agent._should_sanitize_tool_calls = lambda: False
    agent._sanitize_api_messages = lambda value: value
    agent._emit_warning = agent._vprint = lambda *args, **kwargs: None
    agent._drop_thinking_only_and_merge_users = lambda value, **kwargs: value
    agent._extract_reasoning = lambda message: getattr(message, "reasoning", None)
    agent._strip_think_blocks = lambda text: text
    agent.context_compressor = ContextCompressor(
        MODEL, provider=ROUTE[0], base_url=ROUTE[1], api_mode=agent.api_mode,
        quiet_mode=True, config_context_length=200_000,
    )
    agent.context_compressor.bind_session_state(db, "s1")
    return agent


def _patch_assembly_loop(monkeypatch):
    import agent.conversation_loop as loop

    monkeypatch.setattr(loop, "_apply_context_engine_selection", lambda agent, api, history, incoming, logger=None: api)
    monkeypatch.setattr(loop, "_canonicalize_api_tool_calls", lambda messages: None)
    monkeypatch.setattr(loop, "_midturn_request_pressure_tokens", lambda agent, messages, system, approx: approx)
    monkeypatch.setattr(loop, "_pressure_with_real_floor", lambda compressor, value: value)


def _declare(monkeypatch, **model_keys):
    from hermes_cli import config

    cfg = config.load_config()
    cfg["model"] = {"provider": ROUTE[0], "default": MODEL, "supports_vision": True, **model_keys}
    monkeypatch.setattr(config, "load_config_readonly", lambda: cfg)


@pytest.mark.parametrize(("declared", "within"), [(None, 9), (8, 8), (2, 2)])
def test_request_assembly_honours_a_declared_image_limit(tmp_path, monkeypatch, declared, within):
    from agent.turn_request_assembly import assemble_api_request

    _patch_assembly_loop(monkeypatch)
    _declare(monkeypatch, **({"max_images": declared} if declared else {}))
    history = _screenshot_turn(9)
    assembled = assemble_api_request(
        _agent(tmp_path), messages=copy.deepcopy(history), current_turn_user_idx=0,
        _ext_prefetch_cache="", _plugin_user_context="", moa_config=None,
        active_system_prompt="", original_user_message="check the page",
        pending_moa_prepared_request=None, request_logger=logging.getLogger("max-images-test"),
    )

    sent = _image_parts(assembled.api_messages)
    if declared is None:
        assert sent == 9  # the default ceiling lets the whole turn through: the 400 above
    else:
        assert 0 < sent <= within
    assert _image_parts(history) == 9  # only the per-request copy is rewritten
