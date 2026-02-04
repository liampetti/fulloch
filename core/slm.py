"""
Small Language Model module using Qwen via llama.cpp.

Handles loading and running the Qwen language model for intent
detection and conversational AI.
"""

import logging
from typing import Optional

import torch
from llama_cpp import Llama, LlamaGrammar

logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "./data/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"#"./data/models/Qwen3-0.6B-Q4_K_M.gguf"
GRAMMAR_FILE = "./data/models/grammars/json.gbnf"

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
        n_gpu_layers=-1 if DEVICE == "cuda" else 0
    )

    grammar = LlamaGrammar.from_file(grammar_path)

    return grammar, slm_model


def generate_slm(
    slm_model,
    user_prompt: str,
    grammar: Optional[LlamaGrammar] = None,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = N_CONTEXT,
    temperature: float = 0.7,
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

    Returns:
        Generated text response
    """
    # Reset before each call to avoid buffer cache issues
    slm_model.reset()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    stream = slm_model.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        grammar=grammar,
        stream=True,
        temperature=temperature
    )

    full_text = ""
    for chunk in stream:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                token_text = delta["content"]
                full_text += token_text

    return full_text
