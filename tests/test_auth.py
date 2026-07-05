"""Auth helpers — password hashing and session IDs."""

from server.auth import hash_password, new_session_id, verify_password


def test_hash_and_verify_roundtrip():
    pw = "correct-horse-battery-staple"
    assert verify_password(pw, hash_password(pw))


def test_wrong_password_fails():
    assert not verify_password("wrongpass", hash_password("rightpass"))


def test_hash_is_salted():
    assert hash_password("same") != hash_password("same")


def test_malformed_hash_fails_safely():
    assert not verify_password("anything", "no-dollar-sign")


def test_new_session_id_is_url_safe():
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert all(c in valid for c in new_session_id())
