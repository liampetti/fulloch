"""Route-registration facade for satellite transports."""

from typing import TYPE_CHECKING

from fastapi import FastAPI

from .browser_satellite import register_browser_satellite_route
from .satellite_v2 import register_satellite_v2_route

if TYPE_CHECKING:
    from .lifecycle import AppContext, Lifecycle


def register_satellite_routes(app: FastAPI, context: "AppContext", lifecycle: "Lifecycle") -> None:
    """Attach voice-satellite transport endpoints to the application."""
    register_browser_satellite_route(app, context, lifecycle)
    register_satellite_v2_route(app, context, lifecycle)
