"""Task-scoped research capabilities avoid unrelated household history."""

from tools.thinking_playbooks import scoped_capabilities


def test_public_local_model_research_excludes_household_history_tools():
    available = {
        "external_information", "search_papers", "get_entity_history",
        "get_conversation_history", "read_note",
    }

    selected = scoped_capabilities(
        "Research latest local AI models with 9B parameters", available,
    )

    assert selected == {"external_information", "read_note"}


def test_explicit_household_history_request_keeps_history_capabilities():
    available = {"get_entity_history", "get_conversation_history", "read_note"}

    selected = scoped_capabilities("Research my Home Assistant entity history", available)

    assert selected == available


def test_explicit_mixed_model_and_household_research_keeps_requested_history():
    available = {"external_information", "get_entity_history", "read_note"}

    selected = scoped_capabilities(
        "Research local models and compare them with my Home Assistant entity history",
        available,
    )

    assert selected == available
