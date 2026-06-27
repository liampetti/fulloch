"""normalize_url — scheme auto-add + trailing-slash strip for config endpoints."""

from core.url_utils import normalize_url


def test_strips_trailing_slashes():
    assert normalize_url("http://localhost:8123/") == "http://localhost:8123"
    assert normalize_url("http://localhost:8123///") == "http://localhost:8123"
    assert normalize_url("http://host:8080/search/") == "http://host:8080/search"


def test_adds_http_scheme_when_missing():
    assert normalize_url("localhost:8123") == "http://localhost:8123"
    assert normalize_url("192.168.1.50:8123/") == "http://192.168.1.50:8123"
    assert normalize_url("host:8080/search") == "http://host:8080/search"


def test_keeps_explicit_scheme():
    assert normalize_url("https://ha.local/") == "https://ha.local"
    assert normalize_url("https://ha.local") == "https://ha.local"


def test_empty_and_none():
    assert normalize_url("") == ""
    assert normalize_url(None) == ""
    assert normalize_url("   ") == ""
