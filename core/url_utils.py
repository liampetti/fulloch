"""URL normalisation for user-entered config endpoints (HA, SearXNG, remote LLM).

Tidies what users type so a missing scheme or a stray trailing slash doesn't
break requests — e.g. "localhost:8123/" -> "http://localhost:8123". Stdlib-only
and import-light, so any module (tools or core) can use it.
"""


def normalize_url(url, default_scheme: str = "http") -> str:
    """Add a scheme if one is missing and strip trailing slashes.

        "localhost:8123/"   -> "http://localhost:8123"
        "https://ha.local/" -> "https://ha.local"
        "" / None           -> "" (the caller supplies its own default)

    A bare host gets `default_scheme` (http — most self-hosted HA/SearXNG/LLM
    endpoints are plain HTTP on the LAN; users type https:// when they need it).
    """
    u = str(url).strip() if url else ""
    if not u:
        return ""
    if "://" not in u:
        u = f"{default_scheme}://{u}"
    return u.rstrip("/")
