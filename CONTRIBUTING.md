# Contributing to Fulloch

Thank you for your interest in contributing to Fulloch! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   # Install special packages (see requirements.txt for details)
   pip install --no-deps git+https://github.com/liampetti/Qwen3-TTS-streaming.git@97da215
   pip install --no-build-isolation --no-deps git+https://github.com/Dao-AILab/flash-attention.git@ef9e6a6
   pip install -e ".[dev]"  # Install dev dependencies
   ```
4. Copy configuration files:
   ```bash
   cp data/config.example.yml data/config.yml
   cp .env.example .env
   ```
5. Edit `data/config.yml` with your settings

## Adding New Tools

Fulloch uses a decorator-based tool registry system. To add a new tool:

### Step 1: Create a new tool file

Create `tools/my_tool.py`. Import config from the shared loader rather than
re-parsing `config.yml`:

```python
"""My new tool description."""

from ._config import config
from .tool_registry import tool

MY_CONFIG = config.get('my_tool', {})


@tool(
    name="my_function",
    description="What this function does (shown to the SLM in the intent prompt)",
    aliases=["alias1", "alias2"],  # Optional alternative names
)
def my_function(param1: str, param2: int = 10) -> str:
    """
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)

    Returns:
        Result message
    """
    return f"Result: {param1}, {param2}"
```

### Step 2: Register the tool for conditional loading

`tools/__init__.py` keeps two dispatch maps. Add your entry to **one** of them —
`_OPTIONAL_TOOLS` if the tool needs a config block, `_ALWAYS_LOAD` if it has no
config dependency:

```python
# Loaded only when the named top-level key is present in data/config.yml.
_OPTIONAL_TOOLS = {
    'home_assistant': 'home_assistant',
    'search': 'search_web',
    'my_tool': 'my_tool',   # config key → module name
}

# Always loaded — no config dependency.
_ALWAYS_LOAD = ['time_tools', 'thinking', 'notes']
```

### Step 3: Add configuration to the example file

Add a commented-out section to `data/config.example.yml`:

```yaml
# =============================================================================
# My Tool
# =============================================================================
# my_tool:
#   setting1: "value1"
#   setting2: 123
```

### Step 4: Add tests

Tool modules no longer carry `__main__` test blocks. Add a test file under
`tests/` (see `tests/test_tool_registry.py` for the registry fixture pattern,
or `tests/test_intent_catch.py` for plain unit-test style).

### Tool naming and alias collisions

The registry rejects duplicate tool names and aliases with a warning — the
first registration wins. If two tools would claim the same name, only the
first to load is active; the second is skipped. Give the second tool a more
specific name, or disable one via config.

## Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

### Logging

Use the standard logging module:

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Tool return values

Tools are dispatched synchronously by `utils/intents.py:handle_action` and must
return a `str` (the spoken/observed result) or `None`. Returning `None`, raising,
or returning a string beginning with a `... question:` sentinel triggers the
agent's replan loop (see the "Unified Agent Loop" section of `CLAUDE.md`). Keep
tool bodies blocking and simple — for I/O use `requests`/file calls directly; the
orchestrator already runs each turn on its own worker thread.

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes and commit with clear messages:
   ```bash
   git commit -m "Add my new feature"
   ```

3. Run tests before submitting:
   ```bash
   pytest tests/
   ```

4. Push to your fork and create a Pull Request

5. Fill out the PR template with:
   - Summary of changes
   - Test plan
   - Any breaking changes

## Reporting Issues

When reporting issues, please include:

- Python version (`python --version`)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output

## Questions?

Feel free to open an issue for questions or discussion about potential contributions.
