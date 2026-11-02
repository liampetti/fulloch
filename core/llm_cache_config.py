"""Shared local llama-server prompt cache configuration."""

DEFAULT_PROMPT_CACHE_MIB = 1024
MAX_PROMPT_CACHE_MIB = 2**31 - 1


def validate_prompt_cache_mib(value: int) -> int:
    """Accept a bounded MiB value, including zero to disable the cache."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_PROMPT_CACHE_MIB:
        raise ValueError("models.llm.prompt_cache_mib must be a non-negative integer in MiB")
    return value
