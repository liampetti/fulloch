"""Small language model (Qwen via llama.cpp) for intent detection and chat."""

import logging
import time
from typing import Callable, Optional

import torch
from llama_cpp import Llama, LlamaGrammar

from .turn_stats import TurnStats

logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "./data/models/Qwen3.5-9B-Q5_K_M.gguf"
GRAMMAR_FILE = "./data/models/grammars/agent.gbnf"

# 10240 (= 20 × N_BATCH) is a VRAM-vs-context compromise on the 16 GB 5060 Ti:
# 16384 fit at model load but OOM'd on the first decode (the KV cache + compute
# buffer for a full 16k context tipped the card over, aborting in
# ggml_cuda_pool_vmm::alloc). 8192 is the known-good pre-bump value; 10k keeps
# more headroom for history while staying clear of the OOM. Mid-turn overflow is
# handled gracefully (see ContextExhaustedError) rather than failing silently.
N_CONTEXT = 10240
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


def load_slm(
    model_path: str = MODEL_PATH,
    grammar_path: str = GRAMMAR_FILE,
    n_ctx: int = N_CONTEXT,
    n_threads: int = N_THREADS,
    n_batch: int = N_BATCH,
):
    """
    Load the Small Language Model and JSON grammar.

    Args:
        model_path: Path to the GGUF model file
        grammar_path: Path to the JSON grammar file
        n_ctx: Context window size
        n_threads: Number of CPU threads
        n_batch: Batch size for inference

    Returns:
        Tuple of (grammar, model)
    """
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
        thinking_mode: Append Qwen3's `/think` directive so the model emits a
            <think>...</think> block. llama-cpp-python doesn't expose
            chat_template_kwargs, so this in-message switch is the only handle.

    Returns the generated text (still includes <think> blocks — strip with
    `core.text_utils.clean_for_tts` before TTS). Raises ContextExhaustedError
    when the assembled prompt won't fit the model's context.
    """
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
        user_content = f"{user_prompt} /think" if thinking_mode else user_prompt
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
