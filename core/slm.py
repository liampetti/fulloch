"""
Small Language Model module using Qwen via llama.cpp.

Handles loading and running the Qwen language model for intent
detection and conversational AI.
"""

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

N_CONTEXT = 8192
N_THREADS = 4
N_BATCH = 512

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    """
    Generate a response from the language model.

    Args:
        slm_model: The loaded Llama model
        user_prompt: User's input text
        grammar: Optional grammar constraint for structured output
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        cancel_check: Optional callable polled before each streamed chunk;
            returning True aborts generation early (returns the partial text
            accumulated so far). Used by barge-in to drop a mid-flight SLM
            response.
        history: Optional list of prior {"role": ..., "content": ...} messages
            inserted between the system prompt and the current user_prompt.
            Used by the chat path to give the SLM short-term conversation
            memory.
        thinking_mode: When True, append the Qwen3 `/think` directive to the
            user message so the model emits a <think>...</think> reasoning
            block before its final answer. llama-cpp-python doesn't expose
            `chat_template_kwargs`, so this in-message switch is the only
            handle we have on Qwen3 thinking mode.

    Returns:
        Generated text response (still includes <think> blocks — strip with
        `core.text_utils.clean_for_tts` before TTS)
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

    stream = slm_model.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        grammar=grammar,
        stream=True,
        temperature=temperature
    )

    t_call = time.monotonic()
    ttft = None
    out_tokens = 0
    full_text = ""
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

    if stats is not None:
        stats.llm_calls += 1
        stats.llm_gen_seconds += time.monotonic() - t_call
        stats.llm_output_tokens += out_tokens
        if ttft is not None and stats.llm_ttft is None:
            stats.llm_ttft = ttft
        try:
            assembled = "\n".join(m.get("content") or "" for m in messages)
            stats.llm_prompt_tokens += len(slm_model.tokenize(assembled.encode("utf-8")))
        except Exception as e:
            logger.debug(f"Prompt token count failed: {e}")

    return full_text
