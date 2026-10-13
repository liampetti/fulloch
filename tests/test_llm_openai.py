"""OpenAI-compatible remote LLM backend and graceful degradation."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("openai")
pytest.importorskip("httpx")
import httpx  # noqa: E402

from core import llm_openai  # noqa: E402
from core.slm import ContextExhaustedError, RemoteUnreachable, generate_slm  # noqa: E402
from core.turn_stats import TurnStats  # noqa: E402

# --- fakes ------------------------------------------------------------------


def _chunk(content=None, usage=None):
    choices = (
        [types.SimpleNamespace(delta=types.SimpleNamespace(content=content))]
        if content is not None
        else []
    )
    return types.SimpleNamespace(choices=choices, usage=usage)


class _FakeCompletions:
    def __init__(self, behavior):
        self.behavior = behavior
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.behavior(kwargs)


def _make_client(behavior):
    c = llm_openai.OpenAIClient(model="m", base_url="http://x/v1", api_key="k")
    comp = _FakeCompletions(behavior)
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
    return c, comp


# --- generate ---------------------------------------------------------------


def test_streams_and_records_stats():
    usage = types.SimpleNamespace(completion_tokens=2, prompt_tokens=5)
    c, _ = _make_client(lambda k: iter([_chunk("Hel"), _chunk("lo"), _chunk(None, usage=usage)]))
    stats = TurnStats()
    out = c.generate(user_prompt="hi", system_prompt="sys", stats=stats)
    assert out == "Hello"
    assert stats.llm_calls == 1
    assert stats.llm_output_tokens == 2
    assert stats.llm_prompt_tokens == 5
    assert stats.llm_ttft is not None


def test_json_mode_sends_gbnf_grammar():
    # The real agent.gbnf ships in the repo, so json_mode should send it
    # verbatim as a llama.cpp-extension `grammar` field rather than
    # response_format — llama-server silently drops a custom grammar whenever
    # response_format is also present (see core/llm_openai.py's generate()).
    c, comp = _make_client(lambda k: iter([_chunk('{"reply":"hi"}')]))
    out = c.generate(user_prompt="hi", grammar=llm_openai.AGENT_JSON_SENTINEL)
    assert out == '{"reply":"hi"}'
    assert "response_format" not in comp.last_kwargs
    assert "root ::=" in comp.last_kwargs["extra_body"]["grammar"]


def test_response_schema_uses_schema_mode_without_changing_agent_grammar():
    c, comp = _make_client(lambda _kwargs: iter([_chunk('{"actions":[]}')]))
    schema = {"type": "object", "properties": {"actions": {"type": "array"}},
              "required": ["actions"], "additionalProperties": False}
    out = c.generate(user_prompt="plan", response_schema=schema, output_policy="research")
    assert out == '{"actions":[]}'
    assert comp.last_kwargs["response_format"]["type"] == "json_schema"
    assert comp.last_kwargs["response_format"]["json_schema"]["schema"] == schema
    assert "grammar" not in comp.last_kwargs["extra_body"]


def test_response_schema_falls_back_once_to_json_object(monkeypatch):
    from openai import BadRequestError

    calls = []

    def behavior(kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise BadRequestError("unsupported schema",
                                   response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                                   body={})
        return iter([_chunk('{"ok":true}')])

    client, completions = _make_client(behavior)
    response = client.generate(user_prompt="verify", response_schema={"type": "object"})
    assert response == '{"ok":true}'
    assert len(calls) == 2
    assert calls[1]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("emission,finish,expected", [
    ('{"answers":[]', "stop", '{"answers":[]'),
    ('{"answers":oops}', "stop", '{"answers":oops}'),
    ('{"answers":[]}', "length", ""),
    ('{"answers":[]', "length", ""),
])
def test_research_sdk_never_repairs_or_closes_emissions(emission, finish, expected):
    from openai import OpenAI

    requests = []

    def handle(request):
        requests.append(request)
        chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                 "model": "fixture-model", "choices": [{"index": 0,
                 "delta": {"content": emission}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 32, "completion_tokens": 12, "total_tokens": 44}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    c = llm_openai.OpenAIClient("fixture-model", "http://fixture/v1", "fixture-key")
    c._client = OpenAI(base_url=c.base_url, api_key="fixture-key", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    restarted = []
    c._fulloch_restart_local_server = restarted.append
    schema = {"type": "object", "required": ["answers"]}
    diagnostics = {}
    try:
        assert c.generate(user_prompt="fixture", response_schema=schema,
                          output_policy="research", thinking_mode=False, read_timeout=17,
                          generation_timeout=19, recover_on_failure=False,
                          diagnostics=diagnostics) == expected
        assert len(requests) == 1
        body = json.loads(requests[0].content)
        assert body["response_format"]["json_schema"]["schema"] == schema
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert requests[0].extensions["timeout"]["read"] == 17
        assert diagnostics["finish_reason"] == finish
        assert diagnostics["completion_tokens"] == 12
        assert restarted == []
    finally:
        c._client.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_research_sdk_timeout_and_cancellation_never_restart_or_repair(cancelled):
    from openai import OpenAI

    calls = []

    def handle(request):
        calls.append(request)
        if not cancelled:
            raise httpx.ReadTimeout("controlled fixture timeout", request=request)
        chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                 "model": "fixture-model", "choices": [{"index": 0,
                 "delta": {"content": '{"answers":oops}'}, "finish_reason": "stop"}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    c = llm_openai.OpenAIClient("fixture-model", "http://fixture/v1", "fixture-key")
    c._client = OpenAI(base_url=c.base_url, api_key="fixture-key", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    restarted = []
    c._fulloch_restart_local_server = restarted.append
    c._fulloch_local_process = types.SimpleNamespace(poll=lambda: None)
    diagnostics = {}
    try:
        kwargs = {"user_prompt": "fixture", "response_schema": {"type": "object"},
                  "output_policy": "research", "cancel_check": lambda: cancelled,
                  "recover_on_failure": False, "diagnostics": diagnostics}
        if cancelled:
            assert c.generate(**kwargs) == ""
        else:
            with pytest.raises(RemoteUnreachable):
                c.generate(**kwargs)
        assert len(calls) == 1
        assert diagnostics["termination"] == ("cancelled" if cancelled else "timed_out")
        assert restarted == []
    finally:
        c._client.close()


@pytest.mark.parametrize("dead,recover,expected_restart", [(False, True, False), (True, False, False), (True, True, True)])
def test_sdk_original_repair_timeout_chain_retains_separate_usage_and_recovery(dead, recover, expected_restart):
    from openai import OpenAI

    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 2:
            raise httpx.ReadTimeout("controlled repair timeout", request=request)
        chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                 "model": "reported-fixture", "system_fingerprint": "fixture-build",
                 "choices": [{"index": 0, "delta": {"content": '{"reply":oops}'}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 32, "completion_tokens": 12, "total_tokens": 44}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    c = llm_openai.OpenAIClient("requested-fixture", "http://fixture/v1", "fixture-key")
    c._client = OpenAI(base_url=c.base_url, api_key="fixture-key", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    c.context_tokens = 2048
    c._fulloch_local_process = types.SimpleNamespace(poll=lambda: -9 if dead else None, _fulloch_build="fixture-binary")
    restarted = []
    c._fulloch_restart_local_server = restarted.append
    diagnostics = {}
    stats = TurnStats()
    try:
        with pytest.raises(RemoteUnreachable):
            c.generate(user_prompt="private request omitted from diagnostics", grammar=llm_openai.AGENT_JSON_SENTINEL,
                       recover_on_failure=recover, diagnostics=diagnostics, stats=stats)
        original, repair = diagnostics["transport_requests"]
        assert original["completion_tokens"] == 12 and original["finish_reason"] == "stop"
        assert repair["kind"] == "client_format_repair" and repair["termination"] == "timed_out"
        assert repair["error_category"] == "APITimeoutError"
        assert isinstance(original["elapsed_ms"], int) and isinstance(repair["elapsed_ms"], int)
        assert diagnostics["reported_model"] == "reported-fixture"
        assert diagnostics["server_build"] == "fixture-build"
        assert diagnostics["local_build"] == "fixture-binary" and diagnostics["context_tokens"] == 2048
        assert bool(restarted) is expected_restart
        assert bool(repair.get("restarted")) is expected_restart
        assert stats.llm_calls == 2 and stats.llm_output_tokens == 12 and stats.llm_prompt_tokens == 32
        assert "private request" not in str(diagnostics) and "fixture-key" not in str(diagnostics)
    finally:
        c._client.close()


def test_failure_emission_capture_is_explicit_bounded_and_never_contains_request_text(caplog):
    emission = '{"fixture_failure":' + "x" * 9000
    c, _ = _make_client(lambda _kwargs: iter([_chunk(emission)]))
    captures = []
    diagnostics = {}
    kwargs = {"user_prompt": "private household request", "output_policy": "research",
              "response_schema": {"type": "object"}, "diagnostics": diagnostics}
    assert c.generate(**kwargs) == emission
    assert "fixture_failure" not in str(diagnostics)
    assert c.generate(**kwargs, failure_capture=captures.append) == emission
    assert len(captures) == 1 and len(captures[0]["emission"]) == 8192
    assert captures[0]["capture_clipped"] is True
    assert "private household" not in str(captures) and "fixture_failure" not in caplog.text
    # Capture failures must neither crash the job nor disclose the sink exception.
    def broken_sink(_capture):
        raise RuntimeError("private sink detail")

    assert c.generate(**kwargs, failure_capture=broken_sink) == emission
    assert "private sink detail" not in caplog.text


def test_json_mode_falls_back_to_response_format_without_grammar_file(monkeypatch):
    monkeypatch.setattr(llm_openai, "_load_gbnf", lambda: None)
    c, comp = _make_client(lambda k: iter([_chunk('{"reply":"hi"}')]))
    c.generate(user_prompt="hi", grammar=llm_openai.AGENT_JSON_SENTINEL)
    assert comp.last_kwargs["response_format"] == {"type": "json_object"}
    assert "grammar" not in comp.last_kwargs["extra_body"]


def test_free_text_has_no_response_format():
    c, comp = _make_client(lambda k: iter([_chunk("plain text")]))
    c.generate(user_prompt="hi")  # grammar None
    assert "response_format" not in comp.last_kwargs


def test_cancel_check_aborts_before_content():
    c, _ = _make_client(lambda k: iter([_chunk("a"), _chunk("b")]))
    assert c.generate(user_prompt="hi", cancel_check=lambda: True) == ""


def test_cancelled_stream_is_closed_immediately():
    class Stream:
        closed = False

        def __iter__(self):
            return iter([_chunk("a")])

        def close(self):
            self.closed = True

    stream = Stream()
    c, _ = _make_client(lambda _kwargs: stream)

    assert c.generate(user_prompt="hi", cancel_check=lambda: True) == ""
    assert stream.closed is True


def test_local_generation_deadline_does_not_restart_a_healthy_server():
    c, _ = _make_client(lambda k: iter([_chunk("too late")]))
    c._generation_timeout = 0
    restarted = []
    c._fulloch_restart_local_server = restarted.append

    with pytest.raises(RemoteUnreachable, match="deadline"):
        c.generate(user_prompt="hi")

    assert restarted == []


def test_local_generation_deadline_restarts_a_dead_child():
    c, _ = _make_client(lambda _kwargs: iter([_chunk("too late")]))
    c._generation_timeout = 0
    restarted = []
    c._fulloch_restart_local_server = restarted.append
    c._fulloch_local_process = types.SimpleNamespace(poll=lambda: -9)

    with pytest.raises(RemoteUnreachable, match="timed out and was restarted"):
        c.generate(user_prompt="hi")

    assert len(restarted) == 1


def test_generation_deadline_can_preserve_a_shared_local_server():
    c, _ = _make_client(lambda k: iter([_chunk("too late")]))
    c._generation_timeout = 0
    restarted = []
    c._fulloch_restart_local_server = restarted.append

    with pytest.raises(RemoteUnreachable, match="generation exceeded"):
        c.generate(user_prompt="hi", recover_on_failure=False)

    assert restarted == []


def test_local_stream_failure_restarts_the_server():
    def broken_stream(_):
        def stream():
            raise RuntimeError("CUDA worker stopped")
            yield  # pragma: no cover - make this a generator

        return stream()

    c, _ = _make_client(broken_stream)
    restarted = []
    c._fulloch_restart_local_server = restarted.append

    with pytest.raises(RemoteUnreachable, match="failed and was restarted"):
        c.generate(user_prompt="hi")

    assert restarted == ["RuntimeError: CUDA worker stopped"]


def test_max_tokens_clamped_to_reply_ceiling():
    # The N_CONTEXT-sized default must not reach the remote model: no grammar
    # bounds it there, so it would let one answer run thousands of tokens.
    c, comp = _make_client(lambda k: iter([_chunk("x")]))
    c.generate(user_prompt="hi", max_new_tokens=10240)
    assert comp.last_kwargs["max_tokens"] == llm_openai.REMOTE_REPLY_MAX_TOKENS


def test_max_tokens_thinking_uses_roomier_ceiling():
    c, comp = _make_client(lambda k: iter([_chunk("x")]))
    c.generate(user_prompt="ponder", thinking_mode=True, max_new_tokens=10240)
    assert comp.last_kwargs["max_tokens"] == llm_openai.REMOTE_THINK_MAX_TOKENS


def test_smaller_caller_request_is_honoured():
    # The 256-token web summariser must stay 256, not be bumped to the ceiling.
    c, comp = _make_client(lambda k: iter([_chunk("x")]))
    c.generate(user_prompt="hi", max_new_tokens=256)
    assert comp.last_kwargs["max_tokens"] == 256


def test_research_output_is_independent_of_reasoning_mode():
    c, comp = _make_client(lambda k: iter([_chunk("visible")]))
    diagnostics = {}
    assert c.generate(user_prompt="write", max_new_tokens=4096, thinking_mode=False,
                      output_policy="research", diagnostics=diagnostics) == "visible"
    assert comp.last_kwargs["max_tokens"] == diagnostics["effective_output_tokens"] == 4096
    assert diagnostics["termination"] == "completed"


def test_reasoning_only_truncation_records_counts_not_hidden_text():
    usage = types.SimpleNamespace(completion_tokens=2048, prompt_tokens=20,
                                  completion_tokens_details=types.SimpleNamespace(reasoning_tokens=2048))
    chunk = types.SimpleNamespace(choices=[types.SimpleNamespace(
        finish_reason="length", delta=types.SimpleNamespace(content=None, reasoning_content="secret"))], usage=usage)
    c, _ = _make_client(lambda k: iter([chunk]))
    diagnostics = {}
    assert c.generate(user_prompt="write", thinking_mode=True, diagnostics=diagnostics) == ""
    assert diagnostics["termination"] == "truncated"
    assert diagnostics["reasoning_tokens"] == 2048
    assert "secret" not in str(diagnostics)


def test_failed_generation_still_returns_diagnostics():
    c, _ = _make_client(lambda k: (_ for _ in ()).throw(RuntimeError("broken")))
    diagnostics = {}
    with pytest.raises(RemoteUnreachable):
        c.generate(diagnostics=diagnostics)
    assert diagnostics["termination"] == "failed"
    assert diagnostics["error_category"] == "RuntimeError"


def test_thinking_controlled_via_chat_template_kwarg():
    # Thinking is toggled via the chat template's enable_thinking kwarg (clean,
    # no prompt pollution), NOT a /think or /no_think string in the messages —
    # the text switch collided with server-side reasoning control and broke JSON.
    c, comp = _make_client(lambda k: iter([_chunk("x")]))
    c.generate(user_prompt="ponder", system_prompt="sys", thinking_mode=True)
    assert comp.last_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    for m in comp.last_kwargs["messages"]:
        assert "/think" not in m["content"] and "/no_think" not in m["content"]


def test_non_thinking_disables_via_chat_template_kwarg():
    c, comp = _make_client(lambda k: iter([_chunk("x")]))
    c.generate(user_prompt="hi", system_prompt="sys", thinking_mode=False)
    assert comp.last_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    # System prompt is untouched (no directive appended).
    assert comp.last_kwargs["messages"][0]["content"] == "sys"


def test_connect_failure_raises_remote_unreachable():
    from openai import APIConnectionError

    def boom(k):
        raise APIConnectionError(request=httpx.Request("POST", "http://x/v1"))

    c, _ = _make_client(boom)
    with pytest.raises(RemoteUnreachable):
        c.generate(user_prompt="hi")


def test_midstream_failure_with_no_text_raises_remote_unreachable():
    # A failure *during* streaming that isn't connect/timeout/bad-request (the
    # server dropping the connection after the 200, a decode error, etc.) must
    # degrade like an unreachable endpoint rather than crash the turn.
    def gen(k):
        def it():
            raise httpx.RemoteProtocolError("peer closed connection")
            yield  # pragma: no cover — make it a generator

        return it()

    c, _ = _make_client(gen)
    with pytest.raises(RemoteUnreachable):
        c.generate(user_prompt="hi")


def test_500_context_error_raises_context_exhausted():
    """llama.cpp reports a decode context overflow as an HTTP 500."""
    c, _ = _make_client(lambda k: (_ for _ in ()).throw(Exception("Context size has been exceeded")))
    with pytest.raises(ContextExhaustedError):
        c.generate(user_prompt="hi")


def test_midstream_failure_after_partial_returns_partial():
    # If we already streamed some content, keep it instead of discarding the turn.
    def gen(k):
        def it():
            yield _chunk("Partial ")
            yield _chunk("answer")
            raise httpx.RemoteProtocolError("peer closed connection")

        return it()

    c, _ = _make_client(gen)
    assert c.generate(user_prompt="hi") == "Partial answer"


def test_prose_in_json_mode_skips_repair_roundtrip():
    # Plain prose (no '{') isn't an attempt at JSON — the repair would just
    # return prose again, so it must be skipped (only the one streaming call).
    calls = []

    def behavior(kwargs):
        calls.append(kwargs)
        return iter([_chunk("I couldn't find that.")])

    c, _ = _make_client(behavior)
    out = c.generate(user_prompt="hi", grammar=llm_openai.AGENT_JSON_SENTINEL)
    assert out == "I couldn't find that."
    assert len(calls) == 1  # no wasted repair round-trip for prose


def test_malformed_json_still_attempts_repair():
    # Genuinely malformed JSON (has a '{' but isn't fixable by appending
    # closing brackets) is worth one repair round-trip. A truncated-but-
    # closable response is handled by the local fast-path and skips repair.
    calls = []

    def behavior(kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return iter([_chunk('{"reply": hi}')])  # unquoted value -> not closeable
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"reply":"hi"}'))]
        )

    c, _ = _make_client(behavior)
    out = c.generate(user_prompt="hi", grammar=llm_openai.AGENT_JSON_SENTINEL)
    assert len(calls) == 2  # repair was attempted
    assert out == '{"reply":"hi"}'


def test_local_json_repair_5xx_records_and_restarts():
    class _ServerError(RuntimeError):
        status_code = 500

    calls = []

    def behavior(_):
        if not calls:
            calls.append("stream")
            return iter([_chunk('{"reply": hi}')])
        raise _ServerError("decode worker stopped")

    c, _ = _make_client(behavior)
    restarted = []
    c._fulloch_restart_local_server = restarted.append

    with pytest.raises(RemoteUnreachable, match="during JSON repair"):
        c.generate(user_prompt="hi", grammar=llm_openai.AGENT_JSON_SENTINEL)

    assert restarted == ["JSON repair _ServerError: decode worker stopped"]


def test_llama_loading_503_does_not_trigger_another_restart():
    class _Loading(RuntimeError):
        status_code = 503

    def behavior(_):
        def stream():
            raise _Loading("Loading model")
            yield
        return stream()

    c, _ = _make_client(behavior)
    restarted = []
    c._fulloch_restart_local_server = restarted.append

    with pytest.raises(RemoteUnreachable, match="still loading"):
        c.generate(user_prompt="hi")

    assert restarted == []


def test_is_context_error_mapping():
    assert llm_openai._is_context_error(Exception("maximum context length exceeded"))
    assert llm_openai._is_context_error(Exception("This model's context window is 4096"))
    assert llm_openai._is_context_error(Exception("decode() failed: Context size has been exceeded."))
    assert not llm_openai._is_context_error(Exception("rate limit reached"))


def test_generate_slm_dispatches_to_remote():
    class Remote:
        _fulloch_remote = True

        def generate(self, **kw):
            return "REMOTE:" + kw["user_prompt"]

    assert generate_slm(Remote(), user_prompt="x", grammar=object()) == "REMOTE:x"


def test_load_openai_defaults_model_when_unset():
    # model is optional — blank falls back to DEFAULT_MODEL (single-model
    # servers ignore the field; only multi-model endpoints need a real name).
    grammar, client = llm_openai.load_openai(model=None, base_url="http://x/v1")
    assert grammar is llm_openai.AGENT_JSON_SENTINEL
    assert client.model == llm_openai.DEFAULT_MODEL


def test_load_openai_returns_sentinel_and_client():
    grammar, client = llm_openai.load_openai(model="m", base_url="http://x/v1")
    assert grammar is llm_openai.AGENT_JSON_SENTINEL
    assert client._fulloch_remote is True


# --- test_connection --------------------------------------------------------


def test_connection_ok(monkeypatch):
    class _OK:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: object())
            )

    monkeypatch.setattr("openai.OpenAI", _OK)
    assert llm_openai.test_connection("http://x/v1", "m")["ok"] is True


def test_connection_failure(monkeypatch):
    class _Bad:
        def __init__(self, **kw):
            def _raise(**k):
                raise RuntimeError("refused")

            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_raise))

    monkeypatch.setattr("openai.OpenAI", _Bad)
    out = llm_openai.test_connection("http://x/v1", "m")
    assert out["ok"] is False and "refused" in out["error"]


def test_connection_blank_key_falls_back_to_env(monkeypatch):
    # A blank api_key should pick up LLM_API_KEY (mirrors runtime), so the
    # wizard's Test connection works when the key lives only in the environment.
    monkeypatch.setenv("LLM_API_KEY", "envkey")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    class _OK:
        def __init__(self, **kw):
            seen["api_key"] = kw.get("api_key")
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: object())
            )

    monkeypatch.setattr("openai.OpenAI", _OK)
    assert llm_openai.test_connection("http://x/v1", "m", api_key="")["ok"] is True
    assert seen["api_key"] == "envkey"


# --- set_model + list_models ------------------------------------------------


def test_set_model_swaps_model_used_per_request():
    c, comp = _make_client(lambda kw: iter([_chunk("x")]))
    c.set_model("new-model")
    assert c.model == "new-model"
    c.generate(user_prompt="hi")
    assert comp.last_kwargs["model"] == "new-model"


def test_list_models_returns_sorted_ids(monkeypatch):
    class _OK:
        def __init__(self, **kw):
            data = [types.SimpleNamespace(id="zeta"), types.SimpleNamespace(id="alpha")]
            self.models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(data=data))

    monkeypatch.setattr("openai.OpenAI", _OK)
    out = llm_openai.list_models("http://x/v1")
    assert out["ok"] is True and out["models"] == ["alpha", "zeta"]


def test_list_models_failure_degrades(monkeypatch):
    class _Bad:
        def __init__(self, **kw):
            def _raise():
                raise RuntimeError("no endpoint")

            self.models = types.SimpleNamespace(list=_raise)

    monkeypatch.setattr("openai.OpenAI", _Bad)
    out = llm_openai.list_models("http://x/v1")
    assert out["ok"] is False and out["models"] == [] and "no endpoint" in out["error"]


# --- degrade path (agent loop) ---------------------------------------------


def test_agent_loop_degrades_to_regex_on_remote_unreachable():
    import core.agent_loop as al

    calls = {"no_ai_fallback": 0, "llm_error_fallback": 0, "remote_status": []}
    host = types.SimpleNamespace(
        llm_enabled=True,
        _history=[],
        _history_for=lambda session: [],
        _turn_local=types.SimpleNamespace(sink=None, tts_active_event=None),
        grammar=object(),
        wakeword_name="Fulloch",
        tts_session=None,
        replan_stall_cache=[],  # empty -> progress watchdog is a no-op
        play_chunks=lambda *a, **k: None,
        _compact_completed_turns=lambda: None,
        _trim_history=lambda: None,
        _emit_agent_event=lambda *a, **k: None,
        _note_llm_remote_status=lambda ok, error="": calls["remote_status"].append(ok),
    )

    def _raise(**k):
        raise RemoteUnreachable("down")

    host._generate_with_context_recovery = _raise

    def _no_ai_fallback(session, source, satellite_id=None):
        calls["no_ai_fallback"] += 1
        return "BASIC COMMANDS ONLY"

    def _llm_error_fallback(session, source, satellite_id=None):
        calls["llm_error_fallback"] += 1
        return "LLM SERVER UNREACHABLE"

    host._speak_no_ai_fallback = _no_ai_fallback
    host._speak_llm_error_fallback = _llm_error_fallback

    loop = al.AgentLoop(host, session=None, source="text")
    # A prompt the regex fast-path won't catch -> first agent call hits the SLM.
    out = loop.run("tell me a story about the sea")
    # RemoteUnreachable with no regex match -> LLM error fallback, not generic no-AI.
    assert out == "LLM SERVER UNREACHABLE"
    assert calls["llm_error_fallback"] == 1
    assert calls["no_ai_fallback"] == 0
    # The unreachable endpoint was recorded as down (drives the dashboard banner).
    assert calls["remote_status"] == [False]


def test_remote_outage_voice_notice_is_once_per_outage_episode():
    import threading

    from core.assistant import Assistant

    assistant = Assistant.__new__(Assistant)
    assistant.llm_backend = "openai"
    assistant._llm_remote_ok = None
    assistant._llm_remote_error = ""
    assistant._llm_remote_retry_at = 0.0
    assistant._llm_remote_outage_announced = False
    assistant._llm_remote_lock = threading.Lock()
    assistant._speak_llm_error_fallback = lambda *args, **kwargs: "REMOTE UNAVAILABLE"

    assistant._note_llm_remote_status(False, "connection refused")
    assert assistant._remote_llm_retry_blocked() is True
    assert assistant._remote_llm_unavailable_fallback(None, "voice") == "REMOTE UNAVAILABLE"
    assert assistant._remote_llm_unavailable_fallback(None, "voice") == ""
    # Text remains visible and does not consume the voice notice.
    assert assistant._remote_llm_unavailable_fallback(None, "text") == "REMOTE UNAVAILABLE"

    assistant._note_llm_remote_status(True)
    assert assistant._remote_llm_retry_blocked() is False
    assistant._note_llm_remote_status(False, "connection refused")
    assert assistant._remote_llm_unavailable_fallback(None, "voice") == "REMOTE UNAVAILABLE"


# --- token counting and server GPU metrics -----------------------------------


def _client_with_probe(behavior):
    """A client whose chat completions answer the max_tokens=1 count probe."""
    c = llm_openai.OpenAIClient(model="m", base_url="http://x/v1", api_key="k")
    comp = _FakeCompletions(behavior)
    inner = types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
    inner.with_options = lambda **_kw: inner
    c._client = inner
    return c, comp


def test_count_message_tokens_uses_a_one_token_probe_and_reads_usage():
    calls = []

    def behavior(kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            usage=types.SimpleNamespace(prompt_tokens=42, completion_tokens=1))

    c, comp = _client_with_probe(behavior)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert c.count_message_tokens(messages) == 42
    assert c.count_message_tokens(messages, thinking_mode=True) == 42
    # The probe must be a real completion so the server applies its chat
    # template, clamped to a single output token.
    for call in calls:
        assert call["max_tokens"] == 1
        assert call["temperature"] == 0.0
        assert call["messages"] == messages
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[1]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    # An endpoint that does not report usage raises so the budget's conservative
    # estimator stays the fallback instead of a silent undercount.
    c, _ = _client_with_probe(lambda k: types.SimpleNamespace(usage=None))
    with pytest.raises(ValueError):
        c.count_message_tokens(messages)
    with pytest.raises(ValueError):
        c.count_message_tokens([])


def test_server_gpu_memory_metrics_relay_only_the_bounded_counters(monkeypatch):
    c, _ = _make_client(lambda k: None)
    # Without the bundled-server flag nothing is fetched at all.
    assert c.server_gpu_memory_metrics() == {}
    c._fulloch_local_server = True

    c._fulloch_local_process = types.SimpleNamespace(pid=123, poll=lambda: None)

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 1.0
        return types.SimpleNamespace(stdout="123, 100\n456, 9000\n123, 200\n")

    monkeypatch.setattr(llm_openai.subprocess, "run", run)
    assert c.server_gpu_memory_metrics() == {"gpu_allocated_bytes": 300 * 1024 * 1024}

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(llm_openai.subprocess, "run", unavailable)
    assert c.server_gpu_memory_metrics() == {}
