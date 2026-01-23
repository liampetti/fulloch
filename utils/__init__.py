"""
Utilities package for Fulloch voice assistant.

Contains:
- intent_catch: Regex-based intent detection for fast command matching
- intents: Intent handler using the tool registry
- system_prompts: System prompts for AI models
"""

from .intent_catch import catchAll
from .intents import handle_intent, intent_handler
from .system_prompts import (
    getIntentSystemPrompt,
    getChatSystemPrompt,
    getPlannerSystemPrompt,
    getWebSummaryPrompt,
)

__all__ = [
    "catchAll",
    "handle_intent",
    "intent_handler",
    "getIntentSystemPrompt",
    "getChatSystemPrompt",
    "getPlannerSystemPrompt",
    "getWebSummaryPrompt",
]
