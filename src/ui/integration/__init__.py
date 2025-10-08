"""
Integration Module for PDT Trading Bot Admin UI

This package provides integration points between the admin UI
and the existing bot core and compliance modules.
"""

__version__ = "1.0.0"
__author__ = "PDT Trading Bot Team"

from . import bot_core_integration, compliance_integration

__all__ = ["bot_core_integration", "compliance_integration"]
