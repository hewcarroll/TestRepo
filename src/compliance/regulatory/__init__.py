"""
Regulatory Compliance Module

Handles SEC regulations, market manipulation detection, and wash trading prevention.
"""

from .sec_compliance import SECComplianceChecker
from .manipulation_detector import ManipulationDetector
from .wash_trade_detector import WashTradeDetector

__all__ = ['SECComplianceChecker', 'ManipulationDetector', 'WashTradeDetector']