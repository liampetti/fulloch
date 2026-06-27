"""Small language model (Qwen via llama.cpp) for intent detection and chat."""

# Deferred annotations so `LlamaGrammar` in signatures isn't evaluated at import
# (llama_cpp is imported lazily in load_slm, so the CPU image — which has no
# llama-cpp — can still `import core.slm`).
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from llama_cpp import LlamaGrammar

from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "./data/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"
GRAMMAR_FILE = "./data/models/grammars/agent.gbnf"

# 12288 (= 24 × N_BATCH) on the 16 GB 5060 Ti. The default UD-Q4_K_XL quant
# (~6 GB) is ~0.6 GB smaller than the old Q5_K_M, freeing KV-cache headroom, so
# the context went 10240 -> 12288. 16384 historically OOM'd on the first decode
# (KV cache + compute buffer tipped the card over in ggml_cuda_pool_vmm::alloc),
# so that's the rough ceiling — tune per card via models.llm.n_context. Mid-turn
# overflow degrades gracefully (see ContextExhaustedError).
N_CONTEXT = 12288
N_THREADS = 4
N_BATCH = 512

# Tokens held back from n_ctx for the model's own reply plus chat-template
# overhead the naive token estimate doesn't capture. If the assembled prompt
# leaves less than this headroom, the turn can't make progress, so we surface
# a typed error instead of letting llama.cpp fail opaquely mid-eval.
CONTEXT_RESERVE_TOKENS = 512

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ContextExhaustedError(RuntimeError):
    """The assembled prompt no longer fits the model's context window.

    Raised by `generate_slm` (proactively from a token estimate, or as a
    backstop around llama.cpp's own overflow error) so the orchestrator can
    apologise to the user and reset conversation history instead of failing
    silently.
    """


class RemoteUnreachable(RuntimeError):
    """A remote (OpenAI-compatible) LLM endpoint couldn't be reached/used.

    Raised by the remote client on a connect failure (or a read timeout before
    any content). The agent loop catches it and degrades the turn to the
    regex-only no-LLM bypass — no perceptible latency cost on the happy path,
    one failed LAN connect when the endpoint is down. Defined here (not in
    `core.llm_openai`) so the agent loop can import it without pulling the
    openai/httpx stack.
    """


def load_slm(
    model_path: str = MODEL_PATH,
    grammar_path: str = GRAMMAR_FILE,
    n_ctx: int = N_CONTEXT,
    n_threads: int = N_THREADS,
    n_batch: int = N_BATCH,
    think_style: str = "qwen",
):
    """
    Load the Small Language Model and JSON grammar.

    Args:
        model_path: Path to the GGUF model file
        grammar_path: Path to the JSON grammar file
        n_ctx: Context window size
        n_threads: Number of CPU threads
        n_batch: Batch size for inference
        think_style: Reasoning-directive family for this model (set from the
            backend registry). "qwen" appends `/think` in thinking_mode; "" (or
            any other family, e.g. Gemma) adds no directive — see generate_slm.

    Returns:
        Tuple of (grammar, model)
    """
    # Imported here (not at module top) so the CPU image without llama-cpp can
    # still import core.slm — this path only runs when the local llama backend
    # is actually selected.
    from llama_cpp import Llama, LlamaGrammar

    logger.info(f"Loading {model_path} on {DEVICE}...")

    slm_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=-1 if DEVICE == "cuda" else 0,
        # Suppress the "prefix-match"/llama_perf prints on every generate; the
        # dashboard stats panel reports timings now.
        verbose=False,
    )

    grammar = LlamaGrammar.from_file(grammar_path)

    # Stamp the reasoning-directive family so generate_slm picks the right
    # thinking_mode handle without re-consulting the registry (mirrors the
    # `_fulloch_remote` marker on the OpenAI client).
    slm_model._fulloch_think_style = think_style

    return grammar, slm_model


def generate_slm(
    slm_model,
    user_prompt: Optional[str] = None,
    grammar: Optional[LlamaGrammar] = None,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = N_CONTEXT,
    temperature: float = 0.7,
    cancel_check: Optional[Callable[[], bool]] = None,
    history: Optional[list] = None,
    thinking_mode: bool = False,
    stats: Optional[TurnStats] = None,
) -> str:
    """Generate a response from the language model.

    Most args are self-explanatory; the non-obvious ones:
        cancel_check: Polled before each streamed chunk; returning True aborts
            early and returns the partial text. Barge-in uses this to drop a
            mid-flight response.
        history: Prior {"role", "content"} messages inserted between the system
            prompt and user_prompt — the SLM's short-term conversation memory.
            user_prompt is optional so the agent loop can re-call with history
            alone, continuing an in-flight conversation.
        thinking_mode: Request a reasoning turn. The handle depends on the
            model family (stamped as `_fulloch_think_style` by load_slm):
            Qwen3 gets its `/think` text directive appended (emits a
            <think>...</think> block); other families (e.g. Gemma) get no
            directive. llama-cpp-python's create_chat_completion doesn't expose
            chat_template_kwargs, so an in-message text switch is the only handle
            — and Gemma's template can't be toggled that way (it force-closes an
            empty thought channel in the generation prompt), so Gemma deep_think
            runs as a plain considered answer locally. The remote OpenAI backend
            toggles reasoning cleanly via chat_template_kwargs (see llm_openai).

    Returns the generated text (still includes <think> blocks — strip with
    `core.text_utils.clean_for_tts` before TTS). Raises ContextExhaustedError
    when the assembled prompt won't fit the model's context.
    """
    # Remote (OpenAI-compatible) backend: dispatch to its client, which mirrors
    # this signature. `is True` (not just truthy) so a MagicMock fake in tests
    # doesn't accidentally route here. Duck-typed so slm.py doesn't import the
    # openai stack.
    if getattr(slm_model, "_fulloch_remote", False) is True:
        return slm_model.generate(
            user_prompt=user_prompt,
            grammar=grammar,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            cancel_check=cancel_check,
            history=history,
            thinking_mode=thinking_mode,
            stats=stats,
        )

    # Do NOT call slm_model.reset() — llama.cpp's KV-cache prefix matching
    # reuses the prefilled system prompt across consecutive calls. Resetting
    # forces a full reprefill (~520ms on a 1.1k-token intent prompt).
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    # Trailing user message is optional — the agent loop appends tool
    # observations to history and calls generate_slm without a fresh user
    # turn, so the model continues an in-flight conversation.
    if user_prompt is not None:
        # Qwen3 takes an in-message `/think` directive; other families (Gemma)
        # have no working in-process handle, so the text is sent unchanged.
        think_style = getattr(slm_model, "_fulloch_think_style", "qwen")
        if thinking_mode and think_style == "qwen":
            user_content = f"{user_prompt} /think"
        else:
            user_content = user_prompt
        messages.append({"role": "user", "content": user_content})

    # Proactive context-overflow guard. Estimate the assembled prompt size and
    # bail with a typed error before llama.cpp raises an opaque ValueError deep
    # inside create_chat_completion — the caller turns this into a spoken
    # apology + history reset. The estimate (naive content join) matches the
    # stats path below; CONTEXT_RESERVE_TOKENS absorbs the undercount.
    prompt_tokens = None
    try:
        n_ctx = slm_model.n_ctx()
        assembled = "\n".join(m.get("content") or "" for m in messages)
        prompt_tokens = len(slm_model.tokenize(assembled.encode("utf-8")))
        if prompt_tokens > n_ctx - CONTEXT_RESERVE_TOKENS:
            raise ContextExhaustedError(
                f"prompt ~{prompt_tokens} tokens exceeds usable context "
                f"{n_ctx - CONTEXT_RESERVE_TOKENS} (n_ctx={n_ctx})"
            )
    except ContextExhaustedError:
        raise
    except Exception as e:
        logger.debug(f"Context preflight estimate failed: {e}")

    t_call = time.monotonic()
    ttft = None
    out_tokens = 0
    full_text = ""
    try:
        stream = slm_model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            grammar=grammar,
            stream=True,
            temperature=temperature
        )
        for chunk in stream:
            if cancel_check is not None and cancel_check():
                logger.info("SLM generation cancelled")
                break
            choices = chunk.get("choices")
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if content:
                    if ttft is None:
                        ttft = time.monotonic() - t_call
                    out_tokens += 1  # llama.cpp streams ~one token per chunk
                    full_text += content
    except ValueError as e:
        # Backstop: llama.cpp raises a ValueError when the prompt exceeds the
        # context window and the preflight estimate undershot. Re-type it so
        # the orchestrator handles it the same way.
        msg = str(e).lower()
        if "context window" in msg or "exceed" in msg:
            raise ContextExhaustedError(str(e)) from e
        raise

    if stats is not None:
        stats.llm_calls += 1
        stats.llm_gen_seconds += time.monotonic() - t_call
        stats.llm_output_tokens += out_tokens
        if ttft is not None and stats.llm_ttft is None:
            stats.llm_ttft = ttft
        try:
            if prompt_tokens is None:
                assembled = "\n".join(m.get("content") or "" for m in messages)
                prompt_tokens = len(slm_model.tokenize(assembled.encode("utf-8")))
            stats.llm_prompt_tokens += prompt_tokens
        except Exception as e:
            logger.debug(f"Prompt token count failed: {e}")

    return full_text
