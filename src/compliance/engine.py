"""
Compliance Engine

Main compliance coordinator that validates all trading actions before execution.
Ensures strict adherence to SEC regulations, broker policies, and PDT rules.
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from decimal import Decimal
from enum import Enum
from threading import Event
from typing import Dict, List, Optional, Any, Tuple
import json

from .pdt.pdt_tracker import PDTTracker
from .regulatory.sec_compliance import SECComplianceChecker
from .regulatory.manipulation_detector import ManipulationDetector
from .regulatory.wash_trade_detector import WashTradeDetector
from .broker.alpaca_policies import AlpacaPolicyEnforcer
from .audit.audit_logger import AuditLogger


class ComplianceStatus(Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    status: ComplianceStatus
    rule_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requires_approval: bool = False


@dataclass
class ComplianceConfig:
    """Configuration for compliance engine."""
    enable_strict_mode: bool = True  # Hard enforcement of all rules
    enable_audit_logging: bool = True
    max_warnings_before_block: int = 3
    cooling_off_period_minutes: int = 15
    account_balance_threshold: Decimal = Decimal('25000')  # PDT threshold


class ComplianceEngine:
    """
    Main compliance coordinator for trading operations.

    Validates all trading actions against regulatory requirements,
    broker policies, and risk management rules before execution.
    """

    def __init__(self, config: ComplianceConfig, shutdown_event: Optional[Event] = None):
        self.config = config
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("compliance_engine")

        # Initialize compliance modules
        self.pdt_tracker = PDTTracker()
        self.sec_checker = SECComplianceChecker()
        self.manipulation_detector = ManipulationDetector()
        self.wash_trade_detector = WashTradeDetector()
        self.alpaca_enforcer = AlpacaPolicyEnforcer()
        self.audit_logger = AuditLogger()

        # Compliance state tracking
        self.violation_count: Dict[str, int] = {}
        self.warning_count: Dict[str, int] = {}
        self.last_violation_time: Dict[str, datetime] = {}
        self.blocked_until: Dict[str, datetime] = {}

        # Hard-coded safety limits (cannot be modified)
        self._MAX_ORDER_SIZE = Decimal('100000')  # Maximum order value
        self._MAX_DAILY_TRADES = 3  # PDT rule - cannot exceed
        self._MIN_ACCOUNT_BALANCE = Decimal('25000')  # PDT threshold
        self._MAX_POSITION_PCT = Decimal('0.05')  # 5% max position size

        self.logger.info("Compliance engine initialized",
                        strict_mode=config.enable_strict_mode,
                        audit_logging=config.enable_audit_logging)

    async def initialize(self):
        """Initialize compliance engine and all sub-modules."""
        try:
            await self.pdt_tracker.initialize()
            await self.audit_logger.initialize()

            self.logger.info("Compliance engine initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize compliance engine", error=str(e))
            raise

    async def shutdown(self):
        """Shutdown compliance engine and cleanup resources."""
        try:
            await self.audit_logger.shutdown()
            self.logger.info("Compliance engine shutdown complete")
        except Exception as e:
            self.logger.error("Error during compliance engine shutdown", error=str(e))

    async def validate_order(self,
                           symbol: str,
                           action: str,
                           quantity: int,
                           price: Optional[Decimal] = None,
                           account_balance: Optional[Decimal] = None,
                           current_positions: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[ComplianceResult]]:
        """
        Validate a trading order against all compliance rules.

        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Order price (optional)
            account_balance: Current account balance
            current_positions: Current portfolio positions

        Returns:
            Tuple of (is_approved, compliance_results)
        """
        if self.shutdown_event.is_set():
            return False, [ComplianceResult(
                ComplianceStatus.ERROR,
                "SYSTEM_SHUTDOWN",
                "System is shutting down - no new orders allowed"
            )]

        results = []
        order_value = Decimal(str(quantity)) * (price or Decimal('0'))

        try:
            # 1. Basic order validation (hard-coded safety checks)
            basic_result = self._validate_basic_order(symbol, action, quantity, price, order_value)
            results.append(basic_result)
            if basic_result.status == ComplianceStatus.VIOLATION:
                return False, results

            # 2. PDT compliance check
            pdt_result = await self._check_pdt_compliance(action, quantity, account_balance)
            results.append(pdt_result)
            if pdt_result.status == ComplianceStatus.VIOLATION:
                return False, results

            # 3. SEC regulatory compliance
            sec_result = await self._check_sec_compliance(symbol, action, quantity, price)
            results.append(sec_result)
            if sec_result.status == ComplianceStatus.VIOLATION:
                return False, results

            # 4. Market manipulation detection
            manipulation_result = await self._check_manipulation(symbol, action, quantity, price)
            results.append(manipulation_result)
            if manipulation_result.status == ComplianceStatus.VIOLATION:
                return False, results

            # 5. Wash trading detection
            wash_result = await self._check_wash_trading(symbol, action, quantity, current_positions)
            results.append(wash_result)
            if wash_result.status == ComplianceStatus.VIOLATION:
                return False, results

            # 6. Broker policy compliance (Alpaca)
            broker_result = await self._check_broker_policies(symbol, action, quantity, price)
            results.append(broker_result)
            if broker_result.status == ComplianceStatus.VIOLATION:
                return False, results

            # 7. Risk-based validation
            risk_result = self._validate_risk_limits(order_value, account_balance, current_positions)
            results.append(risk_result)
            if risk_result.status == ComplianceStatus.VIOLATION:
                return False, results

            # Check for warnings that might require blocking
            warning_count = sum(1 for r in results if r.status == ComplianceStatus.WARNING)
            if warning_count >= self.config.max_warnings_before_block:
                return False, results

            # Log successful validation
            if self.config.enable_audit_logging:
                await self._log_compliance_decision(symbol, action, quantity, price, True, results)

            return True, results

        except Exception as e:
            error_result = ComplianceResult(
                ComplianceStatus.ERROR,
                "COMPLIANCE_ENGINE_ERROR",
                f"Error during compliance validation: {str(e)}"
            )
            results.append(error_result)

            self.logger.error("Compliance validation error",
                            symbol=symbol,
                            action=action,
                            error=str(e))

            return False, results

    def _validate_basic_order(self, symbol: str, action: str, quantity: int,
                            price: Optional[Decimal], order_value: Decimal) -> ComplianceResult:
        """Hard-coded basic order validation."""

        # Symbol validation
        if not symbol or len(symbol) < 1 or len(symbol) > 5:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "INVALID_SYMBOL",
                f"Invalid symbol format: {symbol}"
            )

        # Action validation
        if action not in ['buy', 'sell']:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "INVALID_ACTION",
                f"Invalid action: {action}. Must be 'buy' or 'sell'"
            )

        # Quantity validation
        if quantity <= 0 or quantity > 1000000:  # Hard-coded max quantity
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "INVALID_QUANTITY",
                f"Invalid quantity: {quantity}. Must be 1-1,000,000"
            )

        # Price validation
        if price is not None and price <= 0:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "INVALID_PRICE",
                f"Invalid price: {price}. Must be positive"
            )

        # Order value validation (hard-coded maximum)
        if order_value > self._MAX_ORDER_SIZE:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "ORDER_VALUE_TOO_LARGE",
                f"Order value ${order_value} exceeds maximum allowed ${self._MAX_ORDER_SIZE}"
            )

        return ComplianceResult(
            ComplianceStatus.COMPLIANT,
            "BASIC_VALIDATION",
            "Basic order validation passed"
        )

    async def _check_pdt_compliance(self, action: str, quantity: int,
                                   account_balance: Optional[Decimal]) -> ComplianceResult:
        """Check PDT rule compliance."""

        # Check if this would be a day trade
        is_day_trade = await self.pdt_tracker.is_day_trade(action, quantity)

        if is_day_trade:
            # Check if account is under PDT threshold
            if account_balance and account_balance < self._MIN_ACCOUNT_BALANCE:
                # Check if we've exceeded the 3 day trade limit
                day_trade_count = await self.pdt_tracker.get_day_trade_count()

                if day_trade_count >= self._MAX_DAILY_TRADES:
                    return ComplianceResult(
                        ComplianceStatus.VIOLATION,
                        "PDT_LIMIT_EXCEEDED",
                        f"Day trade #{day_trade_count + 1} would exceed PDT limit of {self._MAX_DAILY_TRADES} trades in 5 days",
                        details={
                            'current_count': day_trade_count,
                            'limit': self._MAX_DAILY_TRADES,
                            'account_balance': float(account_balance),
                            'threshold': float(self._MIN_ACCOUNT_BALANCE)
                        }
                    )

        # Update PDT tracking
        await self.pdt_tracker.record_trade(action, quantity)

        return ComplianceResult(
            ComplianceStatus.COMPLIANT,
            "PDT_COMPLIANCE",
            "PDT rules satisfied"
        )

    async def _check_sec_compliance(self, symbol: str, action: str, quantity: int,
                                  price: Optional[Decimal]) -> ComplianceResult:
        """Check SEC regulatory compliance."""

        # Use SEC compliance checker
        sec_result = await self.sec_checker.validate_trade(symbol, action, quantity, price)

        if not sec_result['compliant']:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "SEC_VIOLATION",
                sec_result['reason'],
                details=sec_result.get('details', {})
            )

        return ComplianceResult(
            ComplianceStatus.COMPLIANT,
            "SEC_COMPLIANCE",
            "SEC regulations satisfied"
        )

    async def _check_manipulation(self, symbol: str, action: str, quantity: int,
                                price: Optional[Decimal]) -> ComplianceResult:
        """Check for market manipulation patterns."""

        # Use manipulation detector
        manipulation_result = await self.manipulation_detector.detect(symbol, action, quantity, price)

        if manipulation_result['detected']:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "MARKET_MANIPULATION",
                manipulation_result['pattern'],
                details=manipulation_result.get('details', {})
            )

        return ComplianceResult(
            ComplianceStatus.COMPLIANT,
            "MANIPULATION_CHECK",
            "No manipulation patterns detected"
        )

    async def _check_wash_trading(self, symbol: str, action: str, quantity: int,
                                current_positions: Optional[Dict[str, Any]]) -> ComplianceResult:
        """Check for wash trading patterns."""

        # Use wash trade detector
        wash_result = await self.wash_trade_detector.detect(symbol, action, quantity, current_positions)

        if wash_result['detected']:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "WASH_TRADING",
                wash_result['reason'],
                details=wash_result.get('details', {})
            )

        return ComplianceResult(
            ComplianceStatus.COMPLIANT,
            "WASH_TRADE_CHECK",
            "No wash trading patterns detected"
        )

    async def _check_broker_policies(self, symbol: str, action: str, quantity: int,
                                   price: Optional[Decimal]) -> ComplianceResult:
        """Check broker-specific policy compliance."""

        # Use Alpaca policy enforcer
        policy_result = await self.alpaca_enforcer.validate_trade(symbol, action, quantity, price)

        if not policy_result['compliant']:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "BROKER_POLICY_VIOLATION",
                policy_result['reason'],
                details=policy_result.get('details', {})
            )

        return ComplianceResult(
            ComplianceStatus.COMPLIANT,
            "BROKER_POLICY",
            "Broker policies satisfied"
        )

    def _validate_risk_limits(self, order_value: Decimal, account_balance: Optional[Decimal],
                            current_positions: Optional[Dict[str, Any]]) -> ComplianceResult:
        """Validate risk-based limits."""

        if not account_balance:
            return ComplianceResult(
                ComplianceStatus.WARNING,
                "NO_ACCOUNT_BALANCE",
                "Account balance not available for risk validation"
            )

        # Check position size limit (hard-coded 5% max)
        position_pct = order_value / account_balance
        if position_pct > self._MAX_POSITION_PCT:
            return ComplianceResult(
                ComplianceStatus.VIOLATION,
                "POSITION_SIZE_LIMIT",
                f"Order size {position_pct".2%"} exceeds maximum {self._MAX_POSITION_PCT".2%"} of account",
                details={
                    'order_value': float(order_value),
                    'account_balance': float(account_balance),
                    'position_percentage': float(position_pct),
                    'max_percentage': float(self._MAX_POSITION_PCT)
                }
            )

        # Check if order would deplete account
        if order_value > account_balance * Decimal('0.95'):  # 95% max allocation
            return ComplianceResult(
                ComplianceStatus.WARNING,
                "HIGH_ACCOUNT_UTILIZATION",
                f"Order uses {position_pct".1%"} of account balance",
                details={
                    'utilization_percentage': float(position_pct)
                }
            )

        return ComplianceResult(
            ComplianceStatus.COMPLIANT,
            "RISK_LIMITS",
            "Risk limits satisfied"
        )

    async def _log_compliance_decision(self, symbol: str, action: str, quantity: int,
                                     price: Optional[Decimal], approved: bool,
                                     results: List[ComplianceResult]):
        """Log compliance decision for audit trail."""

        decision_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': float(price) if price else None,
            'approved': approved,
            'results': [
                {
                    'rule_name': r.rule_name,
                    'status': r.status.value,
                    'message': r.message,
                    'details': r.details
                }
                for r in results
            ]
        }

        await self.audit_logger.log_compliance_decision(decision_data)

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance engine summary and statistics."""

        return {
            'pdt_status': self.pdt_tracker.get_status(),
            'violation_counts': self.violation_count.copy(),
            'warning_counts': self.warning_count.copy(),
            'blocked_accounts': list(self.blocked_until.keys()),
            'audit_logging_enabled': self.config.enable_audit_logging,
            'strict_mode_enabled': self.config.enable_strict_mode
        }

    def is_account_blocked(self, account_id: str) -> bool:
        """Check if account is currently blocked due to violations."""

        if account_id in self.blocked_until:
            if datetime.now(timezone.utc) < self.blocked_until[account_id]:
                return True
            else:
                # Block period expired
                del self.blocked_until[account_id]

        return False

    def block_account(self, account_id: str, duration_minutes: int = None):
        """Block account due to compliance violations."""

        duration = duration_minutes or self.config.cooling_off_period_minutes
        block_until = datetime.now(timezone.utc).fromtimestamp(
            datetime.now(timezone.utc).timestamp() + (duration * 60)
        )

        self.blocked_until[account_id] = block_until

        self.logger.warning("Account blocked for compliance violations",
                          account_id=account_id,
                          duration_minutes=duration,
                          blocked_until=block_until.isoformat())