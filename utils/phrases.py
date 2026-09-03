"""Short, hand-authored speech used outside LLM replies."""

# Brief conversational receipts for normal agent work. These are deliberately short
ACK_PHRASES = ["Okay.", "Got it.", "Aha.", "Hmm.", "Huh.", "OK.", "Let's see."]

# These operations need no special wording; they reuse the acknowledgement
# clips rather than expanding the startup phrase inventory.
ACK_CACHE_ATTRS = (
    "ack_cache",
    "note_write_stall_cache",
    "pre_thinking_stall_cache",
    "thinking_stall_cache",
    "replan_stall_cache",
)

# Spotify can make several remote calls before audio begins, so its delay is
# worth one explicit lead-in.
MUSIC_SEARCH_PHRASES = ["Finding that.", "Putting that on."]
# Searching can involve slow third-party engines followed by a second model
# call to summarise results, so make its progress explicit rather than saying
# a generic acknowledgement repeatedly.
WEB_SEARCH_PHRASES = ["Looking it up.", "Checking now."]
BUSY_PHRASES = ["I'm helping in another room right now."]
NO_AI_PHRASES = ["I can't do that without an AI model running."]
LLM_ERROR_PHRASES = ["I can't reach the AI server right now. Basic commands still work."]
TOOL_UNAVAILABLE_PHRASES = ["I don't have a tool for that setup yet."]
CONVERSATION_LISTENING_PHRASE = "I'm listening."

# (cache attribute, phrases, LLM-only)
STARTUP_CACHE_SPECS = (
    ("music_search_stall_cache", MUSIC_SEARCH_PHRASES, False),
    ("web_search_stall_cache", WEB_SEARCH_PHRASES, False),
    ("busy_cache", BUSY_PHRASES, False),
    ("no_ai_cache", NO_AI_PHRASES, False),
    ("llm_error_cache", LLM_ERROR_PHRASES, True),
    ("tool_unavailable_cache", TOOL_UNAVAILABLE_PHRASES, True),
    ("conversation_listening_cache", [CONVERSATION_LISTENING_PHRASE], False),
)

REMINDER_PREFIX_PHRASES = ["Just a reminder -", "Heads up -"]

# A compact, varied startup rotation. These topics are used to warm-up the LLM on start-up.
GREETING_TOPICS = [
    "the secret lives of urban wildlife",
    "strange weather phenomena",
    "ancient engineering marvels",
    "the science of everyday smells",
    "how maps changed the world",
    "surprising ways plants communicate",
    "lost languages and undeciphered scripts",
    "the oddest experiments in food history",
    "how modern weather satellites see storms forming",
    "the surprising physics behind roller coasters",
    "recent breakthroughs in protein design",
    "the engineering challenge of reusable rockets",
    "how large language models learn patterns in text",
    "how AI systems learn to recognise images",
    "the challenge of making AI reasoning more reliable",
    "how robots use AI to navigate unfamiliar spaces",
    "the early pioneers of artificial intelligence",
    "the rise and fall of expert systems",
    "how the Turing test shaped AI research",
    "the quest for practical quantum computers",
    "the hidden mathematics of music",
    "how cities are adapting to rising heat",
    "the deep history written in ice cores",
    "why some materials remember their shape",
]
