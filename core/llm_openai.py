"""OpenAI-compatible remote LLM backend with structured-output repair and fallback."""

import json
import logging
import os
import time
from typing import Callable, Optional

import httpx

from .slm import GRAMMAR_FILE, ContextExhaustedError, RemoteUnreachable
from .url_utils import normalize_url

logger = logging.getLogger(__name__)

# Marker object used as the "grammar" for the remote backend: the agent loop
# passes host.grammar (truthy) to request the constrained agent-JSON shape; for
# the remote path its *presence* (not contents) switches on JSON mode.
AGENT_JSON_SENTINEL = object()

_gbnf_text: Optional[str] = None
_gbnf_load_failed = False


def _load_gbnf() -> Optional[str]:
    """Read the local agent grammar once and cache it.

    llama.cpp-family servers (llama-server, Unsloth Studio, LM Studio) accept a
    raw GBNF string per-request via a `grammar` field on `/v1/chat/completions`
    — an undocumented llama.cpp extension, not part of the OpenAI spec — which
    gives the remote path the same hard action/reply-shape enforcement as the
    local GBNF path instead of the looser json_object + repair fallback.
    Servers that don't recognise the field ignore it. Returns None if the file
    can't be read, so the caller degrades to response_format=json_object.
    """
    global _gbnf_text, _gbnf_load_failed
    if _gbnf_text is not None or _gbnf_load_failed:
        return _gbnf_text
    try:
        with open(GRAMMAR_FILE, "r", encoding="utf-8") as f:
            _gbnf_text = f.read()
    except OSError as e:
        logger.warning("Could not read %s for remote grammar passthrough: %s", GRAMMAR_FILE, e)
        _gbnf_load_failed = True
    return _gbnf_text

DEFAULT_BASE_URL = "http://localhost:8888/v1"  # 8080 would clash with SearXNG
# Sent when models.llm.model is unset. Single-model servers (llama-server, a
# one-model LM Studio/Ollama) ignore the field and serve whatever they've
# loaded; only multi-model endpoints (api.openai.com, vLLM with several) need a
# real name.
DEFAULT_MODEL = "default"
DEFAULT_CONNECT_TIMEOUT = 0.4  # seconds — fail fast when the endpoint is down
DEFAULT_READ_TIMEOUT = 30.0  # seconds — enough for queued local inference without a minute of dead air
DEFAULT_GENERATION_TIMEOUT = 90.0  # seconds — bounds a server that continues streaming forever

# Hard output ceilings for the remote path. The local llama.cpp backend is bound
# by the GBNF grammar (it stops at a short JSON emission), so callers pass a huge
# max_new_tokens default (N_CONTEXT) that never bites there. The remote path has
# NO grammar — only response_format=json_object — so that same default is a
# licence to ramble: a "summarise the news" turn was seen generating 8000+ tokens
# (~2 min @ 65 t/s). Disabling reasoning (enable_thinking, below) stops the
# *reasoning* block but not answer length; that's this lever. We clamp every
# remote call to a sane ceiling: a spoken reply
# (or agent JSON) never needs more than ~1k tokens; deep_think (thinking_mode)
# gets a roomier, still-bounded budget since the user explicitly asked for depth.
REMOTE_REPLY_MAX_TOKENS = 1024
REMOTE_THINK_MAX_TOKENS = 8192


def _is_context_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "context length",
            "context window",
            "maximum context",
            "context size has been exceeded",
            "too long",
            "reduce the length",
        )
    )


class OpenAIClient:
    """Thin wrapper over the OpenAI SDK pointed at any compatible endpoint."""

    _fulloch_remote = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        generation_timeout: float = DEFAULT_GENERATION_TIMEOUT,
    ):

        self.model = model
        self.base_url = base_url
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._generation_timeout = generation_timeout
        self._client = self._build_client(api_key)

    def _recover_local_server(self, reason: str) -> bool:
        """Restart the bundled server after a failed local request, if available."""
        restart = getattr(self, "_fulloch_restart_local_server", None)
        if restart is None:
            return False
        record_failure = getattr(self, "_fulloch_record_local_server_failure", None)
        if record_failure is not None:
            try:
                record_failure(reason)
            except Exception:  # noqa: BLE001 - diagnostics must not block recovery
                logger.exception("Could not record local llama-server request failure")
        try:
            restart(reason)
        except Exception:  # noqa: BLE001 - preserve the caller's useful fallback
            logger.exception("Local llama-server recovery failed")
        return True

    def _build_client(self, api_key: str):
        from openai import OpenAI
        timeout = httpx.Timeout(self._read_timeout, connect=self._connect_timeout)
        return OpenAI(
            base_url=self.base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=0,
        )

    def set_model(self, model: str) -> None:
        """Swap the model used for subsequent requests.

        The model is a per-request string (see `generate`'s `model=self.model`),
        not baked into the connection, so this is a cheap live swap — no new
        client, no reconnect. Callers serialise it against in-flight turns.
        """
        self.model = model

    def set_api_key(self, api_key: str) -> None:
        """Swap the API key live — rebuilds the SDK client in-place, no restart needed."""
        self._client = self._build_client(api_key)

    def ping(self) -> tuple[bool, str]:
        """Lightweight reachability probe using this client's own auth + timeout.

        Returns (ok, error). Reuses the configured client so the base_url, API
        key and (fast-fail) connect timeout match the real runtime path — a down
        endpoint, a bad URL, or a non-existent model all come back as ok=False
        with a human-readable error the dashboard can surface. Called off the hot
        path (startup probe), so the one 1-token completion's cost is irrelevant.
        """
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True, ""
        except Exception as e:  # noqa: BLE001 — any failure means "not usable as configured"
            return False, f"{type(e).__name__}: {e}"

    def _messages(self, system_prompt, history, user_prompt):
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        if user_prompt is not None:
            messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate(
        self,
        user_prompt: Optional[str] = None,
        grammar=None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        cancel_check: Optional[Callable[[], bool]] = None,
        history: Optional[list] = None,
        thinking_mode: bool = False,
        stats=None,
        read_timeout: float | None = None,
        generation_timeout: float | None = None,
        recover_on_failure: bool = True,
    ) -> str:
        from openai import APIConnectionError, APITimeoutError, BadRequestError

        messages = self._messages(system_prompt, history, user_prompt)
        json_mode = grammar is not None

        # Clamp to the remote ceiling (see REMOTE_*_MAX_TOKENS): no grammar bounds
        # generation here, so the caller's N_CONTEXT-sized default would otherwise
        # let a single answer run to thousands of tokens. A caller asking for less
        # (e.g. the 256-token web summariser) is honoured.
        ceiling = REMOTE_THINK_MAX_TOKENS if thinking_mode else REMOTE_REPLY_MAX_TOKENS
        max_new_tokens = min(max_new_tokens, ceiling)

        t_call = time.monotonic()
        request_generation_timeout = generation_timeout or self._generation_timeout
        deadline = t_call + request_generation_timeout
        ttft = None
        out_tokens = 0
        full_text = ""
        usage = None
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                # Disable Qwen3 reasoning via the chat template's documented kwarg
                # rather than a /no_think string in the prompt. The text switch
                # collided with server-side reasoning control (e.g. llama-server's
                # --reasoning-budget 0): the model emitted a stray </think> mid-JSON
                # and looped, breaking the emission. chat_template_kwargs renders
                # the template with thinking off cleanly (no prompt pollution).
                # Servers that don't recognise it ignore the extra field.
                "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking_mode}},
            }
            if json_mode:
                gbnf = _load_gbnf()
                if gbnf:
                    # A real grammar is strictly tighter than json_object mode —
                    # and llama-server silently drops a custom grammar in favour
                    # of a trivial schema-derived one whenever response_format
                    # is also present (see common/chat.cpp), so send grammar
                    # alone, never both.
                    kwargs["extra_body"]["grammar"] = gbnf
                else:
                    kwargs["response_format"] = {"type": "json_object"}
            client = self._client
            if read_timeout is not None:
                client = client.with_options(
                    timeout=httpx.Timeout(read_timeout, connect=self._connect_timeout)
                )
            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"LLM generation exceeded the {request_generation_timeout:.0f}s deadline"
                    )
                if cancel_check is not None and cancel_check():
                    logger.info("Remote LLM generation cancelled")
                    break
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if ttft is None:
                        ttft = time.monotonic() - t_call
                    out_tokens += 1
                    full_text += content
        except (APIConnectionError, APITimeoutError) as e:
            local_restarted = recover_on_failure and self._recover_local_server(f"{type(e).__name__}: {e}")
            if local_restarted:
                raise RemoteUnreachable(f"Local llama-server failed and was restarted: {type(e).__name__}: {e}") from e
            # No usable response → degrade. (A read timeout *after* partial
            # content falls through below and returns what we have.)
            if not full_text:
                raise RemoteUnreachable(f"{type(e).__name__}: {e}") from e
        except TimeoutError as e:
            if recover_on_failure and self._recover_local_server(str(e)):
                raise RemoteUnreachable(f"Local llama-server timed out and was restarted: {e}") from e
            raise RemoteUnreachable(str(e)) from e
        except BadRequestError as e:
            if _is_context_error(e):
                raise ContextExhaustedError(str(e)) from e
            raise
        except Exception as e:
            # Any *other* failure mid-stream: the server closed the connection
            # after the 200 (the httpcore "receive_response_body.failed
            # GeneratorExit" symptom), sent a malformed/error SSE chunk, or the
            # SDK raised on decode. The narrow catches above only cover
            # connect/timeout/bad-request, so without this such a failure crashed
            # the whole turn with an opaque error and the user heard nothing.
            # Degrade like an unreachable endpoint: keep any partial text, else
            # raise RemoteUnreachable so the agent loop speaks its fallback. Log
            # the concrete type so the cause is visible next time.
            if _is_context_error(e):
                raise ContextExhaustedError(str(e)) from e
            logger.warning("Remote LLM stream failed mid-response: %s: %s", type(e).__name__, e)
            if recover_on_failure and self._recover_local_server(f"{type(e).__name__}: {e}"):
                raise RemoteUnreachable(f"Local llama-server failed and was restarted: {type(e).__name__}: {e}") from e
            if not full_text:
                raise RemoteUnreachable(f"{type(e).__name__}: {e}") from e

        if json_mode and full_text:
            full_text = self._ensure_json(
                full_text,
                messages,
                max_new_tokens,
                cancel_check=cancel_check,
                deadline=deadline,
            )

        if stats is not None:
            stats.llm_calls += 1
            stats.llm_gen_seconds += time.monotonic() - t_call
            if ttft is not None and stats.llm_ttft is None:
                stats.llm_ttft = ttft
            if usage is not None:
                stats.llm_output_tokens += getattr(usage, "completion_tokens", 0) or out_tokens
                stats.llm_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            else:
                stats.llm_output_tokens += out_tokens

        return full_text

    @staticmethod
    def _close_truncated_json(text: str) -> Optional[str]:
        """Try to close a truncated JSON object by appending the missing brackets.

        Walks the string tracking both brace/bracket depth and string state, then
        appends the closing chars in reverse order. Returns the closed string if
        it parses, else None. Handles the common max_tokens cut-off case where the
        model simply ran out of budget mid-object.
        """
        stack = []
        in_str = False
        escape = False
        for c in text:
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append("}" if c == "{" else "]")
            elif c in "}]" and stack and stack[-1] == c:
                stack.pop()
        if not stack:
            return None  # already balanced — let normal json.loads handle it
        closed = text + "".join(reversed(stack))
        try:
            json.loads(closed)
            return closed
        except Exception:
            return None

    def _ensure_json(
        self,
        text: str,
        messages: list,
        max_new_tokens: int,
        *,
        cancel_check: Optional[Callable[[], bool]] = None,
        deadline: Optional[float] = None,
    ) -> str:
        """Validate the JSON; cheap local close attempt, then one repair round-trip.

        The agent loop already tolerates an unparseable emission, so a failed
        repair degrades to its generic retry rather than crashing.
        """
        try:
            json.loads(text)
            return text
        except Exception:
            pass
        # Fast path: try closing a truncated object (missing closing braces from a
        # max_tokens cut-off) before paying for an LLM repair round-trip.
        closed = self._close_truncated_json(text)
        if closed is not None:
            logger.debug("Remote JSON was truncated; closed locally")
            return closed
        # Only *malformed JSON* (an object that's wrapped or truncated) is worth a
        # repair round-trip. Plain prose with no object delimiter isn't an attempt
        # at JSON — the repair just returns prose again (a wasted ~seconds-long
        # call), and the agent loop recovers prose as a spoken reply anyway. So
        # skip the repair unless there's a `{` to fix.
        if "{" not in text:
            logger.debug("Remote output is prose, not JSON; skipping repair round-trip")
            return text
        if cancel_check is not None and cancel_check():
            logger.info("Skipping JSON repair for cancelled LLM generation")
            return text
        remaining = (deadline - time.monotonic()) if deadline is not None else self._generation_timeout
        if remaining <= 0:
            error = TimeoutError("LLM generation deadline elapsed before JSON repair")
            if self._recover_local_server(str(error)):
                raise RemoteUnreachable(f"Local llama-server timed out and was restarted: {error}") from error
            raise RemoteUnreachable(str(error)) from error
        logger.debug("Remote JSON invalid; attempting one repair")
        try:
            from openai import APIConnectionError, APITimeoutError

            repair = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with ONLY the corrected JSON object.",
                },
            ]
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=repair,
                temperature=0.0,
                max_tokens=max_new_tokens,
                response_format={"type": "json_object"},
                timeout=min(self._read_timeout, remaining),
            )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("LLM generation deadline elapsed during JSON repair")
            if cancel_check is not None and cancel_check():
                logger.info("Discarding JSON repair result for cancelled LLM generation")
                return text
            fixed = resp.choices[0].message.content or text
            json.loads(fixed)
            return fixed
        except (APIConnectionError, APITimeoutError) as e:
            if self._recover_local_server(f"JSON repair {type(e).__name__}: {e}"):
                raise RemoteUnreachable(
                    f"Local llama-server failed and was restarted during JSON repair: {type(e).__name__}: {e}"
                ) from e
            raise RemoteUnreachable(f"{type(e).__name__}: {e}") from e
        except TimeoutError as e:
            if self._recover_local_server(str(e)):
                raise RemoteUnreachable(f"Local llama-server timed out and was restarted: {e}") from e
            raise RemoteUnreachable(str(e)) from e
        except Exception as e:
            # A 5xx from the local server is a server failure even though the
            # original streaming response completed. Preserve diagnostics and
            # recycle it just like a stream disconnect; malformed repair output
            # itself remains a normal parse failure for the agent loop to handle.
            status_code = getattr(e, "status_code", 0)
            if status_code >= 500:
                if self._recover_local_server(f"JSON repair {type(e).__name__}: {e}"):
                    raise RemoteUnreachable(
                        f"Local llama-server failed and was restarted during JSON repair: {type(e).__name__}: {e}"
                    ) from e
            return text  # let the agent loop's parse-failure path handle it


def _resolve_api_key(explicit: str = "") -> str:
    return (
        explicit
        or os.environ.get("LLM_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or "not-needed"
    )


def load_openai(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,  # accepted but ignored — key comes from credentials.json
    connect_timeout=None,
    read_timeout=None,
    **opts,
):
    """Build the remote client. Returns (AGENT_JSON_SENTINEL, OpenAIClient).

    The sentinel stands in for the local GBNF grammar so the agent loop's
    `grammar=host.grammar` call still signals "constrain to agent JSON".
    """
    # model is optional — blank falls back to DEFAULT_MODEL, which single-model
    # servers ignore (only multi-model endpoints need a real name).
    model = model or DEFAULT_MODEL
    # Normalise: add a scheme if missing, drop a trailing slash (which would
    # otherwise produce "…//chat/completions").
    base_url = normalize_url(base_url or DEFAULT_BASE_URL)
    client = OpenAIClient(
        model=model,
        base_url=base_url,
        api_key=_resolve_api_key(),
        connect_timeout=float(connect_timeout) if connect_timeout else DEFAULT_CONNECT_TIMEOUT,
        read_timeout=float(read_timeout) if read_timeout else DEFAULT_READ_TIMEOUT,
        generation_timeout=(
            float(opts["generation_timeout"])
            if opts.get("generation_timeout") is not None
            else DEFAULT_GENERATION_TIMEOUT
        ),
    )
    logger.info("Remote LLM backend: %s @ %s", model, base_url)
    return AGENT_JSON_SENTINEL, client


def test_connection(
    base_url: str,
    model: str,
    api_key: str = "",
    connect_timeout: float = 2.0,
    read_timeout: float = 10.0,
) -> dict:
    """Probe an endpoint with a 1-token completion. Returns {ok, error}.

    Used by the wizard's 'Test connection' button (a deliberate pre-flight,
    unlike the runtime call-and-catch). A slightly looser connect timeout here
    since the user is actively waiting. A blank api_key falls back to the env
    key (LLM_API_KEY / OPENAI_API_KEY) so the test mirrors runtime.
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url,
            api_key=_resolve_api_key(api_key),
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            max_retries=0,
        )
        client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return {"ok": True, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def list_models(
    base_url: str, api_key: str = "", connect_timeout: float = 2.0, read_timeout: float = 10.0
) -> dict:
    """List the models an OpenAI-compatible endpoint advertises (`GET /v1/models`).

    Returns {ok, models (sorted ids), error}. Lets the UI offer a picker after a
    successful test connection instead of free-text entry. Not every server
    implements the endpoint — a failure degrades to {ok: False} and the caller
    falls back to manual entry.
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=normalize_url(base_url),
            api_key=api_key or "not-needed",
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            max_retries=0,
        )
        resp = client.models.list()
        ids = sorted({m.id for m in resp.data if getattr(m, "id", None)})
        return {"ok": True, "models": ids, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "models": [], "error": f"{type(e).__name__}: {e}"}
