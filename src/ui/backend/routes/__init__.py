"""
API Routes for PDT Trading Bot Admin UI

Contains all REST API endpoints for data access, authentication,
and dashboard functionality.
"""

from . import auth, data, dashboard

__all__ = ["auth", "data", "dashboard"]