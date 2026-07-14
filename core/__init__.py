"""Core package.

Intentionally empty: importing `core` must NOT eagerly pull in
`core.assistant` (and its model dependencies), so pure-logic
test files can collect and run on a machine with no GPU stack.
Import the heavy surface explicitly where it's needed:

    from core.assistant import Assistant
"""
