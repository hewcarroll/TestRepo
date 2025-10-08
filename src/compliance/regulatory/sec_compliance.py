"""
SEC Compliance Checker

Validates trades against SEC regulations and ensures regulatory compliance.
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone, time
from decimal import Decimal
from threading import Event
from typing import Dict, List, Optional, Any, Set
import json

from ..audit.audit_logger import AuditLogger


@dataclass
class SECRule:
    """Represents an SEC regulation rule."""
    rule_id: str
    name: str
    description: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class SECComplianceChecker:
    """
    SEC regulatory compliance checker.

    Validates trading activities against SEC regulations including:
    - Regulation NMS (National Market System)
    - Regulation SHO (Short Sale restrictions)
    - Market manipulation rules
    - Insider trading prevention
    """

    def __init__(self, shutdown_event: Optional[Event] = None):
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("sec_compliance")
        self.audit_logger = AuditLogger()

        # Hard-coded SEC rules (cannot be modified at runtime)
        self._rules = {
            'REG_NMS': SECRule(
                rule_id='REG_NMS',
                name='Regulation NMS',
                description='National Market System - Best execution requirements',
                parameters={
                    'max_spread_pct': Decimal('0.05'),  # 5% maximum spread
                    'min_quote_size': 100,  # Minimum quote size
                    'max_order_size': 1000000  # Maximum order size
                }
            ),
            'REG_SHO': SECRule(
                rule_id='REG_SHO',
                name='Regulation SHO',
                description='Short sale restrictions and locate requirements',
                parameters={
                    'require_locate_for_shorts': True,
                    'max_fail_to_deliver_days': 3,
                    'threshold_security_list': set()  # Would be populated from SEC data
                }
            ),
            'MARKET_MANIPULATION': SECRule(
                rule_id='MARKET_MANIPULATION',
                name='Market Manipulation Prevention',
                description='Prevention of manipulative trading practices',
                parameters={
                    'max_position_concentration': Decimal('0.10'),  # 10% max position
                    'min_holding_period': 300,  # 5 minutes minimum hold
                    'max_orders_per_minute': 10
                }
            ),
            'INSIDER_TRADING': SECRule(
                rule_id='INSIDER_TRADING',
                name='Insider Trading Prevention',
                description='Prevention of trading on material non-public information',
                parameters={
                    'restricted_symbols': set(),  # Would be populated from SEC filings
                    'blackout_periods': dict(),  # Company-specific blackout periods
                    'mnpi_monitoring': True
                }
            )
        }

        # Trading activity tracking for compliance
        self._order_history: List[Dict[str, Any]] = []
        self._position_history: Dict[str, List[Dict[str, Any]]] = {}
        self._trade_frequency: Dict[str, List[datetime]] = {}

        self.logger.info("SEC compliance checker initialized",
                        active_rules=len([r for r in self._rules.values() if r.enabled]))

    async def initialize(self):
        """Initialize SEC compliance checker."""
        try:
            await self.audit_logger.initialize()
            await self._load_sec_data()
            self.logger.info("SEC compliance checker initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize SEC compliance checker", error=str(e))
            raise

    async def shutdown(self):
        """Shutdown SEC compliance checker."""
        try:
            await self.audit_logger.shutdown()
            self.logger.info("SEC compliance checker shutdown complete")
        except Exception as e:
            self.logger.error("Error during SEC compliance checker shutdown", error=str(e))

    async def validate_trade(self, symbol: str, action: str, quantity: int,
                           price: Optional[Decimal] = None) -> Dict[str, Any]:
        """
        Validate a trade against SEC regulations.

        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Trade price

        Returns:
            Dict with 'compliant' boolean and 'reason' string
        """
        if self.shutdown_event.is_set():
            return {
                'compliant': False,
                'reason': 'System shutdown - no trading allowed',
                'rule_violations': ['SYSTEM_SHUTDOWN']
            }

        violations = []
        warnings = []

        try:
            # Check Regulation NMS compliance
            nms_result = await self._check_reg_nms(symbol, action, quantity, price)
            if not nms_result['compliant']:
                violations.append(nms_result['violation'])

            # Check Regulation SHO compliance
            sho_result = await self._check_reg_sho(symbol, action, quantity)
            if not sho_result['compliant']:
                violations.append(sho_result['violation'])

            # Check market manipulation rules
            manipulation_result = await self._check_market_manipulation(symbol, action, quantity)
            if not manipulation_result['compliant']:
                violations.append(manipulation_result['violation'])

            # Check insider trading rules
            insider_result = await self._check_insider_trading(symbol, action, quantity)
            if not insider_result['compliant']:
                violations.append(insider_result['violation'])

            # Record trade for compliance monitoring
            await self._record_trade_activity(symbol, action, quantity, price)

            if violations:
                # Log violations for audit
                await self._log_sec_violation(symbol, action, violations)

                return {
                    'compliant': False,
                    'reason': f"SEC violations: {', '.join(violations)}",
                    'rule_violations': violations,
                    'details': {
                        'nms_check': nms_result,
                        'sho_check': sho_result,
                        'manipulation_check': manipulation_result,
                        'insider_check': insider_result
                    }
                }

            return {
                'compliant': True,
                'reason': 'All SEC regulations satisfied',
                'warnings': warnings
            }

        except Exception as e:
            self.logger.error("Error during SEC compliance check",
                            symbol=symbol,
                            action=action,
                            error=str(e))

            return {
                'compliant': False,
                'reason': f'SEC compliance check error: {str(e)}',
                'rule_violations': ['COMPLIANCE_CHECK_ERROR']
            }

    async def _check_reg_nms(self, symbol: str, action: str, quantity: int,
                           price: Optional[Decimal]) -> Dict[str, Any]:
        """Check Regulation NMS compliance."""

        rule = self._rules['REG_NMS']

        # Check order size limits
        if quantity > rule.parameters['max_order_size']:
            return {
                'compliant': False,
                'violation': 'REG_NMS_ORDER_SIZE',
                'reason': f'Order size {quantity} exceeds NMS limit {rule.parameters["max_order_size"]}'
            }

        # Check if price is reasonable (would need market data in real implementation)
        if price and price <= 0:
            return {
                'compliant': False,
                'violation': 'REG_NMS_INVALID_PRICE',
                'reason': f'Invalid price: {price}'
            }

        # Check for best execution requirements (simplified check)
        # In a real implementation, this would check multiple exchanges for best price
        if await self._is_market_manipulation_attempt(symbol, action, quantity, price):
            return {
                'compliant': False,
                'violation': 'REG_NMS_BEST_EXECUTION',
                'reason': 'Potential best execution violation detected'
            }

        return {'compliant': True}

    async def _check_reg_sho(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Check Regulation SHO compliance."""

        rule = self._rules['REG_SHO']

        # Check if this is a short sale
        if action == 'sell':
            # In a real implementation, this would check if the account has the shares
            # For now, we'll assume all sells are potentially short sales
            if rule.parameters['require_locate_for_shorts']:
                # Check if we have a valid locate for this symbol
                if not await self._has_valid_locate(symbol):
                    return {
                        'compliant': False,
                        'violation': 'REG_SHO_NO_LOCATE',
                        'reason': f'No valid locate for short sale of {symbol}'
                    }

        # Check if symbol is on threshold security list
        if symbol in rule.parameters['threshold_security_list']:
            return {
                'compliant': False,
                'violation': 'REG_SHO_THRESHOLD_SECURITY',
                'reason': f'{symbol} is on threshold security list - short sales restricted'
            }

        return {'compliant': True}

    async def _check_market_manipulation(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Check for market manipulation patterns."""

        rule = self._rules['MARKET_MANIPULATION']

        # Check position concentration
        # In a real implementation, this would check actual portfolio positions
        if await self._is_high_concentration(symbol, quantity):
            return {
                'compliant': False,
                'violation': 'MARKET_MANIPULATION_CONCENTRATION',
                'reason': f'Position concentration would exceed {rule.parameters["max_position_concentration"]".1%"}'
            }

        # Check trade frequency (potential layering/spoofing)
        if await self._is_excessive_trading_frequency(symbol):
            return {
                'compliant': False,
                'violation': 'MARKET_MANIPULATION_FREQUENCY',
                'reason': 'Excessive trading frequency detected'
            }

        # Check for rapid order cancellation (potential spoofing)
        if await self._is_spoofing_pattern(symbol, action):
            return {
                'compliant': False,
                'violation': 'MARKET_MANIPULATION_SPOOFING',
                'reason': 'Potential spoofing pattern detected'
            }

        return {'compliant': True}

    async def _check_insider_trading(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Check for insider trading indicators."""

        rule = self._rules['INSIDER_TRADING']

        # Check if symbol is on restricted list
        if symbol in rule.parameters['restricted_symbols']:
            return {
                'compliant': False,
                'violation': 'INSIDER_TRADING_RESTRICTED',
                'reason': f'{symbol} is on restricted trading list'
            }

        # Check if currently in blackout period for this symbol
        if await self._is_blackout_period(symbol):
            return {
                'compliant': False,
                'violation': 'INSIDER_TRADING_BLACKOUT',
                'reason': f'{symbol} is in blackout period'
            }

        # Check for unusual trading patterns that might indicate MNPI
        if await self._is_suspicious_trading_pattern(symbol, action, quantity):
            return {
                'compliant': False,
                'violation': 'INSIDER_TRADING_SUSPICIOUS',
                'reason': 'Suspicious trading pattern detected'
            }

        return {'compliant': True}

    async def _record_trade_activity(self, symbol: str, action: str, quantity: int, price: Optional[Decimal]):
        """Record trade activity for compliance monitoring."""

        trade_record = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': float(price) if price else None
        }

        self._order_history.append(trade_record)

        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self._order_history = [t for t in self._order_history if t['timestamp'] > cutoff_time]

        # Track frequency by symbol
        if symbol not in self._trade_frequency:
            self._trade_frequency[symbol] = []

        self._trade_frequency[symbol].append(trade_record['timestamp'])

        # Keep only recent frequency data (last hour)
        hour_ago = datetime.now(timezone.utc).replace(minute=datetime.now(timezone.utc).minute - 60)
        self._trade_frequency[symbol] = [t for t in self._trade_frequency[symbol] if t > hour_ago]

    async def _has_valid_locate(self, symbol: str) -> bool:
        """Check if we have a valid short locate for the symbol."""
        # In a real implementation, this would check with prime brokers
        # For now, we'll simulate based on symbol characteristics
        return not symbol.startswith('0')  # Simplified check

    async def _is_market_manipulation_attempt(self, symbol: str, action: str, quantity: int, price: Optional[Decimal]) -> bool:
        """Check for potential market manipulation attempts."""
        # Simplified check - in reality this would use sophisticated algorithms
        return False

    async def _is_high_concentration(self, symbol: str, quantity: int) -> bool:
        """Check if position would create high concentration."""
        # Simplified check - in reality this would check actual portfolio
        return False

    async def _is_excessive_trading_frequency(self, symbol: str) -> bool:
        """Check for excessive trading frequency."""

        rule = self._rules['MARKET_MANIPULATION']
        max_orders = rule.parameters['max_orders_per_minute']

        if symbol in self._trade_frequency:
            recent_trades = self._trade_frequency[symbol]
            minute_ago = datetime.now(timezone.utc).replace(second=0, microsecond=0)

            trades_last_minute = [t for t in recent_trades if t > minute_ago]
            return len(trades_last_minute) >= max_orders

        return False

    async def _is_spoofing_pattern(self, symbol: str, action: str) -> bool:
        """Check for potential spoofing patterns."""
        # Simplified check - in reality this would analyze order patterns
        return False

    async def _is_blackout_period(self, symbol: str) -> bool:
        """Check if symbol is in blackout period."""
        # In a real implementation, this would check against earnings calendars, etc.
        return False

    async def _is_suspicious_trading_pattern(self, symbol: str, action: str, quantity: int) -> bool:
        """Check for suspicious trading patterns that might indicate insider trading."""
        # Simplified check - in reality this would use ML models
        return False

    async def _log_sec_violation(self, symbol: str, action: str, violations: List[str]):
        """Log SEC violations for audit trail."""

        violation_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'sec_violation',
            'symbol': symbol,
            'action': action,
            'violations': violations,
            'rule_details': {
                rule_id: {
                    'name': rule.name,
                    'description': rule.description,
                    'parameters': rule.parameters
                }
                for rule_id, rule in self._rules.items()
                if rule.rule_id in violations
            }
        }

        await self.audit_logger.log_compliance_violation(violation_data)

    async def _load_sec_data(self):
        """Load SEC data and regulatory lists."""
        try:
            # In a real implementation, this would load from SEC EDGAR, threshold lists, etc.
            # For now, we'll initialize with empty data
            self.logger.info("SEC regulatory data loaded")
        except Exception as e:
            self.logger.error("Error loading SEC data", error=str(e))

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current SEC compliance status."""

        return {
            'active_rules': len([r for r in self._rules.values() if r.enabled]),
            'total_rules': len(self._rules),
            'recent_orders': len(self._order_history),
            'tracked_symbols': len(self._trade_frequency),
            'rules': {
                rule_id: {
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'description': rule.description
                }
                for rule_id, rule in self._rules.items()
            }
        }