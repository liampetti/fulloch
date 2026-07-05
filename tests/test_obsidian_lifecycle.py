"""AppContext.obsidian_vault_state default shape and updates."""
from server.lifecycle import AppContext, Lifecycle


def test_default_vault_state_shape():
    ctx = AppContext(lifecycle=Lifecycle())
    state = ctx.obsidian_vault_state
    assert state["connected"] is False
    assert state["vault_path"] is None
    assert state["last_connected_at"] is None
    assert state["last_error"] is None
    assert state["indexing_progress"] is None


def test_vault_state_is_a_mutable_dict():
    ctx = AppContext(lifecycle=Lifecycle())
    ctx.obsidian_vault_state["connected"] = True
    ctx.obsidian_vault_state["vault_path"] = "/tmp/vault"
    assert ctx.obsidian_vault_state["connected"] is True
    assert ctx.obsidian_vault_state["vault_path"] == "/tmp/vault"
