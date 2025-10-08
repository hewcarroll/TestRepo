"""
Alpaca Policy Enforcer

Enforces Alpaca-specific trading policies, rate limits, and broker restrictions.
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from threading import Event
from typing import Dict, List, Optional, Any
import json

from ..audit.audit_logger import AuditLogger


@dataclass
class AlpacaPolicy:
    """Represents an Alpaca-specific policy."""
    policy_id: str
    name: str
    description: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class AlpacaPolicyEnforcer:
    """
    Alpaca broker policy enforcement system.
    
    Enforces Alpaca-specific trading policies including:
    - Rate limiting and API usage limits
    - Margin requirements and trading restrictions
    - Market hours and session limits
    - Position limits and order restrictions
    """
    
    def __init__(self, shutdown_event: Optional[Event] = None):
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("alpaca_policies")
        self.audit_logger = AuditLogger()
        
        # Hard-coded Alpaca policy limits (based on Alpaca documentation)
        self._policies = {
            'RATE_LIMIT': AlpacaPolicy(
                policy_id='RATE_LIMIT',
                name='API Rate Limiting',
                description='Alpaca API rate limits and throttling',
                parameters={
                    'orders_per_second': 1,  # Conservative limit
                    'orders_per_minute': 30,
                    'requests_per_minute': 200,
                    'burst_limit': 5
                }
            ),
            'MARGIN_REQUIREMENT': AlpacaPolicy(
                policy_id='MARGIN_REQUIREMENT',
                name='Margin Requirements',
                description='Margin account requirements and restrictions',
                parameters={
                    'min_account_value': Decimal('2000'),
                    'margin_maintenance': Decimal('0.25'),  # 25%
                    'day_trading_margin': Decimal('0.25'),  # 25%
                    'overnight_margin': Decimal('0.50')  # 50%
                }
            ),
            'MARKET_HOURS': AlpacaPolicy(
                policy_id='MARKET_HOURS',
                name='Market Hours Compliance',
                description='Trading session and market hours restrictions',
                parameters={
                    'pre_market_start': '07:00:00',
                    'pre_market_end': '09:30:00',
                    'regular_hours_start': '09:30:00',
                    'regular_hours_end': '16:00:00',
                    'after_hours_start': '16:00:00',
                    'after_hours_end': '20:00:00',
                    'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                    'holidays': []  # Would be populated with market holidays
                }
            ),
            'POSITION_LIMITS': AlpacaPolicy(
                policy_id='POSITION_LIMITS',
                name='Position and Order Limits',
                description='Position sizing and order value restrictions',
                parameters={
                    'max_order_value': Decimal('100000'),
                    'max_position_value': Decimal('500000'),
                    'max_day_orders': 1000,
                    'max_position_pct': Decimal('0.10')  # 10% of portfolio
                }
            ),
            'SHORT_SELLING': AlpacaPolicy(
                policy_id='SHORT_SELLING',
                name='Short Selling Restrictions',
                description='Short sale locate and borrowing requirements',
                parameters={
                    'require_locate': True,
                    'max_short_ratio': Decimal('0.50'),  # 50% of portfolio
                    'locate_timeout': 300  # 5 minutes
                }
            )
        }
        
        # Rate limiting tracking
        self._request_counts: Dict[str, List[datetime]] = {}
        self._order_counts: Dict[str, List[datetime]] = {}
        self._account_info: Optional[Dict[str, Any]] = None
        self._margin_info: Optional[Dict[str, Any]] = None
        
        self.logger.info("Alpaca policy enforcer initialized",
                        active_policies=len([p for p in self._policies.values() if p.enabled]))
    
    async def initialize(self):
        """Initialize Alpaca policy enforcer."""
        try:
            await self.audit_logger.initialize()
            self.logger.info("Alpaca policy enforcer initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize Alpaca policy enforcer", error=str(e))
            raise
    
    async def shutdown(self):
        """Shutdown Alpaca policy enforcer."""
        try:
            await self.audit_logger.shutdown()
            self.logger.info("Alpaca policy enforcer shutdown complete")
        except Exception as e:
            self.logger.error("Error during Alpaca policy enforcer shutdown", error=str(e))
    
    async def validate_trade(self, symbol: str, action: str, quantity: int,
                           price: Optional[Decimal] = None) -> Dict[str, Any]:
        """
        Validate trade against Alpaca policies.
        
        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Trade price
            
        Returns:
            Dict with 'compliant' boolean and policy violations
        """
        if self.shutdown_event.is_set():
            return {
                'compliant': False,
                'reason': 'System shutdown - no trading allowed',
                'policy_violations': ['SYSTEM_SHUTDOWN']
            }
        
        violations = []
        warnings = []
        
        try:
            # Check rate limiting
            rate_result = await self._check_rate_limits()
            if not rate_result['compliant']:
                violations.append(rate_result['violation'])
            
            # Check market hours
            hours_result = await self._check_market_hours()
            if not hours_result['compliant']:
                violations.append(hours_result['violation'])
            
            # Check margin requirements
            margin_result = await self._check_margin_requirements(symbol, action, quantity, price)
            if not margin_result['compliant']:
                violations.append(margin_result['violation'])
            
            # Check position limits
            position_result = await self._check_position_limits(symbol, action, quantity, price)
            if not position_result['compliant']:
                violations.append(position_result['violation'])
            
            # Check short selling rules
            short_result = await self._check_short_selling(symbol, action, quantity)
            if not short_result['compliant']:
                violations.append(short_result['violation'])
            
            if violations:
                # Log violations for audit
                await self._log_policy_violation(symbol, action, violations)
                
                return {
                    'compliant': False,
                    'reason': f"Alpaca policy violations: {', '.join(violations)}",
                    'policy_violations': violations,
                    'details': {
                        'rate_check': rate_result,
                        'hours_check': hours_result,
                        'margin_check': margin_result,
                        'position_check': position_result,
                        'short_check': short_result
                    }
                }
            
            return {
                'compliant': True,
                'reason': 'All Alpaca policies satisfied',
                'warnings': warnings
            }
            
        except Exception as e:
            self.logger.error("Error during Alpaca policy check",
                            symbol=symbol,
                            action=action,
                            error=str(e))
            
            return {
                'compliant': False,
                'reason': f'Alpaca policy check error: {str(e)}',
                'policy_violations': ['POLICY_CHECK_ERROR']
            }
    
    async def _check_rate_limits(self) -> Dict[str, Any]:
        """Check API rate limiting compliance."""
        
        policy = self._policies['RATE_LIMIT']
        now = datetime.now(timezone.utc)
        
        # Check orders per second
        second_ago = now - timedelta(seconds=1)
        recent_orders = [
            t for t in self._order_counts.get('orders', [])
            if t > second_ago
        ]
        
        if len(recent_orders) >= policy.parameters['orders_per_second']:
            return {
                'compliant': False,
                'violation': 'RATE_LIMIT_ORDERS_PER_SECOND',
                'reason': f'Order rate limit exceeded: {len(recent_orders)} orders in 1 second'
            }
        
        # Check orders per minute
        minute_ago = now - timedelta(minutes=1)
        orders_last_minute = [
            t for t in self._order_counts.get('orders', [])
            if t > minute_ago
        ]
        
        if len(orders_last_minute) >= policy.parameters['orders_per_minute']:
            return {
                'compliant': False,
                'violation': 'RATE_LIMIT_ORDERS_PER_MINUTE',
                'reason': f'Order rate limit exceeded: {len(orders_last_minute)} orders in 1 minute'
            }
        
        return {'compliant': True}
    
    async def _check_market_hours(self) -> Dict[str, Any]:
        """Check market hours compliance."""
        
        policy = self._policies['MARKET_HOURS']
        now = datetime.now(timezone.utc)
        current_time = now.time()
        
        # Check if today is a trading day
        today_name = now.strftime('%A')
        if today_name not in policy.parameters['trading_days']:
            return {
                'compliant': False,
                'violation': 'MARKET_HOURS_NOT_TRADING_DAY',
                'reason': f'{today_name} is not a trading day'
            }
        
        # Check if current time is within trading hours
        pre_market_start = datetime.strptime(policy.parameters['pre_market_start'], '%H:%M:%S').time()
        pre_market_end = datetime.strptime(policy.parameters['pre_market_end'], '%H:%M:%S').time()
        regular_start = datetime.strptime(policy.parameters['regular_hours_start'], '%H:%M:%S').time()
        regular_end = datetime.strptime(policy.parameters['regular_hours_end'], '%H:%M:%S').time()
        after_hours_start = datetime.strptime(policy.parameters['after_hours_start'], '%H:%M:%S').time()
        after_hours_end = datetime.strptime(policy.parameters['after_hours_end'], '%H:%M:%S').time()
        
        in_trading_hours = False
        
        if pre_market_start <= current_time <= pre_market_end:
            in_trading_hours = True
        elif regular_start <= current_time <= regular_end:
            in_trading_hours = True
        elif after_hours_start <= current_time <= after_hours_end:
            in_trading_hours = True
        
        if not in_trading_hours:
            return {
                'compliant': False,
                'violation': 'MARKET_HOURS_OUTSIDE_SESSION',
                'reason': f'Current time {current_time} is outside trading hours'
            }
        
        return {'compliant': True}
    
    async def _check_margin_requirements(self, symbol: str, action: str, quantity: int,
                                       price: Optional[Decimal]) -> Dict[str, Any]:
        """Check margin account requirements."""
        
        policy = self._policies['MARGIN_REQUIREMENT']
        
        if not self._account_info:
            return {
                'compliant': False,
                'violation': 'MARGIN_INFO_UNAVAILABLE',
                'reason': 'Account information not available for margin check'
            }
        
        # Check minimum account value
        account_value = Decimal(str(self._account_info.get('portfolio_value', '0')))
        if account_value < policy.parameters['min_account_value']:
            return {
                'compliant': False,
                'violation': 'MARGIN_MIN_ACCOUNT_VALUE',
                'reason': f'Account value ${account_value} below minimum ${policy.parameters["min_account_value"]}'
            }
        
        # Check margin utilization for buy orders
        if action == 'buy' and price:
            order_value = Decimal(str(quantity)) * price
            buying_power = Decimal(str(self._account_info.get('buying_power', '0')))
            
            if order_value > buying_power:
                return {
                    'compliant': False,
                    'violation': 'MARGIN_INSUFFICIENT_BUYING_POWER',
                    'reason': f'Order value ${order_value} exceeds buying power ${buying_power}'
                }
        
        return {'compliant': True}
    
    async def _check_position_limits(self, symbol: str, action: str, quantity: int,
                                   price: Optional[Decimal]) -> Dict[str, Any]:
        """Check position and order size limits."""
        
        policy = self._policies['POSITION_LIMITS']
        
        # Check order value limit
        if price:
            order_value = Decimal(str(quantity)) * price
            if order_value > policy.parameters['max_order_value']:
                return {
                    'compliant': False,
                    'violation': 'POSITION_ORDER_VALUE_LIMIT',
                    'reason': f'Order value ${order_value} exceeds limit ${policy.parameters["max_order_value"]}'
                }
        
        # Check position percentage limit (would need portfolio info in real implementation)
        # For now, we'll use a simplified check
        if price and self._account_info:
            order_value = Decimal(str(quantity)) * price
            portfolio_value = Decimal(str(self._account_info.get('portfolio_value', '0')))
            
            if portfolio_value > 0:
                position_pct = order_value / portfolio_value
                if position_pct > policy.parameters['max_position_pct']:
                    return {
                        'compliant': False,
                        'violation': 'POSITION_PERCENTAGE_LIMIT',
                        'reason': f'Position would be {position_pct:.1%} of portfolio, exceeds {policy.parameters["max_position_pct"]:.1%} limit'
                    }
        
        return {'compliant': True}
    
    async def _check_short_selling(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Check short selling policy compliance."""
        
        policy = self._policies['SHORT_SELLING']
        
        if action == 'sell':
            # Check if we need a locate for this short sale
            if policy.parameters['require_locate']:
                # In a real implementation, this would check locate status
                # For now, we'll assume locate is required for all shorts
                if not await self._has_valid_locate(symbol):
                    return {
                        'compliant': False,
                        'violation': 'SHORT_SELLING_NO_LOCATE',
                        'reason': f'No valid locate for short sale of {symbol}'
                    }
        
        return {'compliant': True}
    
    async def _has_valid_locate(self, symbol: str) -> bool:
        """Check if we have a valid short locate for the symbol."""
        # In a real implementation, this would check with Alpaca's locate system
        # For now, we'll use a simplified check
        return symbol not in ['HTZ', 'GME']  # Example restricted symbols
    
    async def _log_policy_violation(self, symbol: str, action: str, violations: List[str]):
        """Log policy violations for audit trail."""
        
        violation_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'alpaca_policy_violation',
            'symbol': symbol,
            'action': action,
            'violations': violations,
            'policy_details': {
                policy_id: {
                    'name': policy.name,
                    'description': policy.description,
                    'parameters': policy.parameters
                }
                for policy_id, policy in self._policies.items()
                if policy.policy_id in violations
            }
        }
        
        await self.audit_logger.log_compliance_violation(violation_data)
    
    def record_api_request(self, request_type: str = 'other'):
        """Record API request for rate limiting."""
        now = datetime.now(timezone.utc)
        
        if request_type not in self._request_counts:
            self._request_counts[request_type] = []
        
        self._request_counts[request_type].append(now)
        
        # Clean old requests (keep last minute)
        minute_ago = now - timedelta(minutes=1)
        self._request_counts[request_type] = [
            t for t in self._request_counts[request_type]
            if t > minute_ago
        ]
    
    def record_order(self):
        """Record order for rate limiting."""
        now = datetime.now(timezone.utc)
        
        if 'orders' not in self._order_counts:
            self._order_counts['orders'] = []
        
        self._order_counts['orders'].append(now)
        
        # Clean old orders (keep last minute)
        minute_ago = now - timedelta(minutes=1)
        self._order_counts['orders'] = [
            t for t in self._order_counts['orders']
            if t > minute_ago
        ]
    
    def update_account_info(self, account_info: Dict[str, Any]):
        """Update account information for policy checks."""
        self._account_info = account_info
    
    def update_margin_info(self, margin_info: Dict[str, Any]):
        """Update margin information for policy checks."""
        self._margin_info = margin_info
    
    def get_policy_status(self) -> Dict[str, Any]:
        """Get current policy enforcement status."""
        
        return {
            'active_policies': len([p for p in self._policies.values() if p.enabled]),
            'total_policies': len(self._policies),
            'rate_limit_status': {
                'orders_per_second': len(self._order_counts.get('orders', [])),
                'requests_per_minute': sum(len(reqs) for reqs in self._request_counts.values())
            },
            'account_status': {
                'has_account_info': self._account_info is not None,
                'has_margin_info': self._margin_info is not None
            },
            'policies': {
                policy_id: {
                    'name': policy.name,
                    'enabled': policy.enabled,
                    'description': policy.description
                }
                for policy_id, policy in self._policies.items()
            }
        }
