"""
FastAPI Backend for PDT Trading Bot Admin UI

Provides REST API endpoints, WebSocket connections, and authentication
for the admin dashboard interface.
"""

from .app import app

__all__ = ["app"]