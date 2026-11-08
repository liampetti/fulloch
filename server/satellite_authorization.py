"""Authorization for sensitive satellite provisioning HTTP requests."""

import time
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from .auth import COOKIE_MAX_AGE, SESSION_COOKIE


def authorize(request: Request, context) -> None:
    """Require an active password session and same-origin, non-simple requests.

    The explicit header is also required of non-browser clients. Browsers cannot
    send it cross-origin without an allowed CORS preflight. No proxy headers are
    trusted here; a reverse proxy must forward the public Host and scheme safely.
    """
    if not context.dashboard_password_hash:
        raise HTTPException(
            status_code=403, detail="Enable a dashboard password before provisioning"
        )
    session_id = request.cookies.get(SESSION_COOKIE, "")
    created = context.sessions.get(session_id) if session_id else None
    now = time.time()
    if (
        not isinstance(created, (int, float))
        or isinstance(created, bool)
        or not 0 <= now - created < COOKIE_MAX_AGE
    ):
        raise HTTPException(
            status_code=401, detail="A valid dashboard password session is required"
        )
    if request.headers.get("x-fulloch-provisioning") != "1":
        raise HTTPException(status_code=403, detail="Missing provisioning request header")
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlsplit(origin)
            request_url = request.url
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.scheme != request_url.scheme
                or parsed.hostname != request_url.hostname
                or (parsed.port or (443 if parsed.scheme == "https" else 80))
                != (request_url.port or (443 if request_url.scheme == "https" else 80))
            ):
                raise ValueError("origin mismatch")
        except ValueError as error:
            raise HTTPException(
                status_code=403, detail="Invalid provisioning request origin"
            ) from error
