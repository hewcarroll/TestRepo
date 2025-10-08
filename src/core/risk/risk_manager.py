"""
Risk Management Module for PDT Trading Bot

Comprehensive risk management system with position limits, stop-loss logic,
daily loss limits, and portfolio diversification controls.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from threading import Event
from typing import Dict, List, Optional, Any, Callable, Tuple
import structlog


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: Decimal = Decimal('0.05')  # 5% max position size
    daily_loss_limit_pct: Decimal = Decimal('0.01')  # 1% daily loss limit
    stop_loss_pct: Decimal = Decimal('0.02')  # 2% stop loss per position
    max_positions: int = 10  # Maximum number of simultaneous positions
    max_sector_exposure: Decimal = Decimal('0.20')  # 20% max per sector
    correlation_limit: float = 0.7  # Maximum correlation between positions

    # PDT-specific limits
    max_daily_trades: int = 50  # Conservative daily trade limit
    min_trade_interval: int = 60  # Minimum seconds between trades for same symbol


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    entry_price: Decimal
    current_price: Decimal
    quantity: int
    stop_loss_price: Decimal
    max_loss_pct: Decimal
    unrealized_pnl: Decimal
    sector: Optional[str] = None
    correlation_group: Optional[str] = None


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_value: Decimal
    total_exposure: Decimal
    daily_pnl: Decimal
    daily_pnl_pct: Decimal
    num_positions: int
    sector_exposure: Dict[str, Decimal]
    largest_position_pct: Decimal
    correlation_risk: float
    leverage_ratio: Decimal = Decimal('1.0')


class RiskEvent:
    """Risk management event."""

    def __init__(self, event_type: str, symbol: Optional[str] = None, **kwargs):
        self.event_type = event_type
        self.symbol = symbol
        self.timestamp = datetime.now(timezone.utc)
        self.data = kwargs


class RiskManager:
    """
    Comprehensive risk management system for PDT trading.

    Monitors position limits, enforces stop-losses, tracks daily loss limits,
    and manages portfolio diversification to minimize risk exposure.
    """

    def __init__(self,
                 max_position_pct: Decimal = Decimal('0.05'),
                 daily_loss_limit_pct: Decimal = Decimal('0.01'),
                 stop_loss_pct: Decimal = Decimal('0.02'),
                 shutdown_event: Optional[Event] = None):
        self.config = RiskConfig(
            max_position_pct=max_position_pct,
            daily_loss_limit_pct=daily_loss_limit_pct,
            stop_loss_pct=stop_loss_pct
        )
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("risk_manager")

        # Risk callback
        self._risk_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # Position tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.account_balance: Optional[Decimal] = None
        self.daily_start_balance: Optional[Decimal] = None

        # Daily tracking
        self.daily_pnl: Decimal = Decimal('0')
        self.daily_trades: int = 0
        self.session_start: Optional[datetime] = None

        # Risk metrics
        self.risk_metrics = PortfolioRisk(
            total_value=Decimal('0'),
            total_exposure=Decimal('0'),
            daily_pnl=Decimal('0'),
            daily_pnl_pct=Decimal('0'),
            num_positions=0,
            sector_exposure={},
            largest_position_pct=Decimal('0'),
            correlation_risk=0.0
        )

        self.logger.info("Risk manager initialized",
                        max_position_pct=float(max_position_pct),
                        daily_loss_limit_pct=float(daily_loss_limit_pct))

    def set_risk_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for risk events."""
        self._risk_callback = callback

    async def initialize(self):
        """Initialize risk management system."""
        self.logger.info("Initializing risk manager")

        # Reset daily tracking for new session
        self.session_start = datetime.now(timezone.utc)
        self.daily_pnl = Decimal('0')
        self.daily_trades = 0

        # Initialize account balance if not set
        if self.account_balance is None:
            self.account_balance = Decimal('25000')  # Default PDT threshold

        self.daily_start_balance = self.account_balance

        self.logger.info("Risk manager initialized",
                        account_balance=float(self.account_balance),
                        session_start=self.session_start.isoformat())

    async def shutdown(self):
        """Shutdown risk manager."""
        self.logger.info("Shutting down risk manager")

        # Clear all positions
        self.positions.clear()

        # Reset metrics
        self.risk_metrics = PortfolioRisk(
            total_value=Decimal('0'),
            total_exposure=Decimal('0'),
            daily_pnl=Decimal('0'),
            daily_pnl_pct=Decimal('0'),
            num_positions=0,
            sector_exposure={},
            largest_position_pct=Decimal('0'),
            correlation_risk=0.0
        )

        self.logger.info("Risk manager shutdown complete")

    async def validate_trade(self, symbol: str, action: str, quantity: int, price: Decimal) -> bool:
        """
        Validate trade against risk limits.

        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Trade price

        Returns:
            bool: True if trade is allowed, False otherwise
        """
        try:
            if self.shutdown_event.is_set():
                return False

            trade_value = Decimal(str(quantity)) * price

            # Check daily trade limit
            if not self._check_daily_trade_limit():
                self.logger.warning("Daily trade limit exceeded")
                await self._emit_risk_event("daily_trade_limit", symbol)
                return False

            # Check position limits
            if action == "buy":
                if not self._check_position_limits(symbol, quantity, price):
                    return False

                if not self._check_portfolio_limits(trade_value):
                    return False

            # Check stop-loss for existing positions
            if action == "sell" and symbol in self.positions:
                if not self._check_stop_loss(symbol, price):
                    return False

            # Check minimum trade interval
            if not self._check_trade_interval(symbol):
                self.logger.warning("Trade interval too short", symbol=symbol)
                return False

            # Update tracking if trade would be valid
            if action == "buy":
                self.daily_trades += 1

            self.logger.debug("Trade validation passed",
                            symbol=symbol,
                            action=action,
                            quantity=quantity,
                            price=float(price))

            return True

        except Exception as e:
            self.logger.error("Error validating trade",
                            symbol=symbol,
                            action=action,
                            error=str(e))
            return False

    def _check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit is exceeded."""
        return self.daily_trades < self.config.max_daily_trades

    def _check_position_limits(self, symbol: str, quantity: int, price: Decimal) -> bool:
        """Check position-specific risk limits."""
        if self.account_balance is None:
            self.logger.error("Account balance not set")
            return False

        trade_value = Decimal(str(quantity)) * price
        max_position_value = self.account_balance * self.config.max_position_pct

        # Check if this position would exceed max position size
        if trade_value > max_position_value:
            self.logger.warning("Position size exceeds limit",
                              symbol=symbol,
                              trade_value=float(trade_value),
                              max_position_value=float(max_position_value))
            return False

        # Check maximum number of positions
        if symbol not in self.positions and len(self.positions) >= self.config.max_positions:
            self.logger.warning("Maximum positions exceeded",
                              current_positions=len(self.positions),
                              max_positions=self.config.max_positions)
            return False

        return True

    def _check_portfolio_limits(self, trade_value: Decimal) -> bool:
        """Check portfolio-level risk limits."""
        if self.account_balance is None:
            return False

        # Check daily loss limit
        if self.daily_start_balance:
            current_balance = self.account_balance + self.daily_pnl
            daily_loss = self.daily_start_balance - current_balance
            daily_loss_pct = daily_loss / self.daily_start_balance

            if daily_loss_pct > self.config.daily_loss_limit_pct:
                self.logger.error("Daily loss limit exceeded",
                                daily_loss_pct=float(daily_loss_pct),
                                limit_pct=float(self.config.daily_loss_limit_pct))
                asyncio.create_task(self._emit_risk_event("daily_loss_limit"))
                return False

        # Check total portfolio exposure
        total_exposure = sum(
            abs(pos.quantity) * pos.current_price
            for pos in self.positions.values()
        )

        new_exposure = total_exposure + trade_value
        max_exposure = self.account_balance * Decimal('2.0')  # 200% max exposure

        if new_exposure > max_exposure:
            self.logger.warning("Portfolio exposure limit exceeded",
                              new_exposure=float(new_exposure),
                              max_exposure=float(max_exposure))
            return False

        return True

    def _check_stop_loss(self, symbol: str, current_price: Decimal) -> bool:
        """Check if position should be stopped out."""
        if symbol not in self.positions:
            return True

        position = self.positions[symbol]
        price_change_pct = (current_price - position.entry_price) / position.entry_price

        # Check if we've hit stop loss
        if price_change_pct <= -self.config.stop_loss_pct:
            self.logger.warning("Stop loss triggered",
                              symbol=symbol,
                              price_change_pct=float(price_change_pct),
                              stop_loss_pct=float(-self.config.stop_loss_pct))

            asyncio.create_task(self._emit_risk_event("stop_loss_triggered", symbol,
                                                    price_change_pct=float(price_change_pct)))
            return False

        return True

    def _check_trade_interval(self, symbol: str) -> bool:
        """Check minimum time between trades for same symbol."""
        # For now, always allow (can be enhanced with symbol-specific tracking)
        return True

    def add_position(self, symbol: str, quantity: int, price: Decimal, sector: str = "UNKNOWN"):
        """Add or update position for risk tracking."""
        try:
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                old_quantity = position.quantity

                # Calculate weighted average price for position sizing
                if old_quantity * quantity > 0:  # Same direction
                    total_quantity = old_quantity + quantity
                    if total_quantity != 0:
                        position.entry_price = (
                            (position.entry_price * Decimal(str(old_quantity)) +
                             price * Decimal(str(quantity))) / Decimal(str(total_quantity))
                        )
                else:  # Opposite direction (partial close)
                    if abs(quantity) < abs(old_quantity):
                        # Partial close - keep existing entry price
                        pass
                    else:
                        # Full close or reversal
                        position.entry_price = price

                position.quantity = quantity
                position.current_price = price
                position.sector = sector

            else:
                # New position
                stop_loss_price = price * (1 - self.config.stop_loss_pct)
                self.positions[symbol] = PositionRisk(
                    symbol=symbol,
                    entry_price=price,
                    current_price=price,
                    quantity=quantity,
                    stop_loss_price=stop_loss_price,
                    max_loss_pct=self.config.stop_loss_pct,
                    unrealized_pnl=Decimal('0'),
                    sector=sector
                )

            # Update risk metrics
            self._update_risk_metrics()

            self.logger.debug("Position added/updated",
                            symbol=symbol,
                            quantity=quantity,
                            price=float(price))

        except Exception as e:
            self.logger.error("Error adding position",
                            symbol=symbol,
                            error=str(e))

    def remove_position(self, symbol: str):
        """Remove position from risk tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            self._update_risk_metrics()
            self.logger.debug("Position removed", symbol=symbol)

    def update_position_price(self, symbol: str, current_price: Decimal):
        """Update current price for position risk calculations."""
        if symbol in self.positions:
            position = self.positions[symbol]

            # Calculate unrealized P&L
            price_change = current_price - position.entry_price
            position.unrealized_pnl = price_change * Decimal(str(position.quantity))
            position.current_price = current_price

            # Check stop-loss
            if not self._check_stop_loss(symbol, current_price):
                # Stop-loss triggered - position should be closed
                asyncio.create_task(self._emit_risk_event("stop_loss_triggered", symbol))

    def _update_risk_metrics(self):
        """Update portfolio risk metrics."""
        try:
            if not self.positions:
                self.risk_metrics = PortfolioRisk(
                    total_value=Decimal('0'),
                    total_exposure=Decimal('0'),
                    daily_pnl=self.daily_pnl,
                    daily_pnl_pct=Decimal('0'),
                    num_positions=0,
                    sector_exposure={},
                    largest_position_pct=Decimal('0'),
                    correlation_risk=0.0
                )
                return

            # Calculate portfolio metrics
            total_exposure = Decimal('0')
            sector_exposure = {}
            largest_position = Decimal('0')

            for position in self.positions.values():
                position_value = abs(position.quantity) * position.current_price
                total_exposure += position_value

                # Track sector exposure
                sector = position.sector or "UNKNOWN"
                sector_exposure[sector] = sector_exposure.get(sector, Decimal('0')) + position_value

                # Track largest position
                position_pct = position_value / self.account_balance if self.account_balance else Decimal('0')
                largest_position = max(largest_position, position_pct)

            # Calculate daily P&L percentage
            daily_pnl_pct = Decimal('0')
            if self.daily_start_balance and self.daily_start_balance > 0:
                current_balance = (self.account_balance or Decimal('0')) + self.daily_pnl
                daily_pnl_pct = self.daily_pnl / self.daily_start_balance

            # Update risk metrics
            self.risk_metrics = PortfolioRisk(
                total_value=self.account_balance or Decimal('0'),
                total_exposure=total_exposure,
                daily_pnl=self.daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                num_positions=len(self.positions),
                sector_exposure=sector_exposure,
                largest_position_pct=largest_position,
                correlation_risk=self._calculate_correlation_risk()
            )

        except Exception as e:
            self.logger.error("Error updating risk metrics", error=str(e))

    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk (simplified)."""
        if len(self.positions) < 2:
            return 0.0

        # Simplified correlation risk based on sector concentration
        if self.risk_metrics.sector_exposure:
            max_sector_pct = max(self.risk_metrics.sector_exposure.values()) / self.risk_metrics.total_exposure
            return float(max_sector_pct)

        return 0.0

    async def _emit_risk_event(self, event_type: str, symbol: Optional[str] = None, **kwargs):
        """Emit risk event to callback."""
        try:
            event_data = {
                'type': event_type,
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                **kwargs,
                'risk_metrics': {
                    'daily_pnl': float(self.risk_metrics.daily_pnl),
                    'daily_pnl_pct': float(self.risk_metrics.daily_pnl_pct),
                    'num_positions': self.risk_metrics.num_positions,
                    'total_exposure': float(self.risk_metrics.total_exposure)
                }
            }

            if self._risk_callback:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._risk_callback, event_data
                )

        except Exception as e:
            self.logger.error("Error emitting risk event",
                            event_type=event_type,
                            error=str(e))

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            'account_balance': float(self.account_balance) if self.account_balance else 0,
            'daily_pnl': float(self.daily_pnl),
            'daily_pnl_pct': float(self.risk_metrics.daily_pnl_pct),
            'daily_trades': self.daily_trades,
            'num_positions': len(self.positions),
            'total_exposure': float(self.risk_metrics.total_exposure),
            'largest_position_pct': float(self.risk_metrics.largest_position_pct),
            'sector_exposure': {k: float(v) for k, v in self.risk_metrics.sector_exposure.items()},
            'correlation_risk': self.risk_metrics.correlation_risk,
            'positions': [
                {
                    'symbol': p.symbol,
                    'quantity': p.quantity,
                    'entry_price': float(p.entry_price),
                    'current_price': float(p.current_price),
                    'unrealized_pnl': float(p.unrealized_pnl),
                    'stop_loss_price': float(p.stop_loss_price),
                    'sector': p.sector
                }
                for p in self.positions.values()
            ],
            'config': {
                'max_position_pct': float(self.config.max_position_pct),
                'daily_loss_limit_pct': float(self.config.daily_loss_limit_pct),
                'stop_loss_pct': float(self.config.stop_loss_pct),
                'max_positions': self.config.max_positions
            }
        }

    def set_account_balance(self, balance: Decimal):
        """Set current account balance."""
        self.account_balance = balance
        if self.daily_start_balance is None:
            self.daily_start_balance = balance
        self._update_risk_metrics()

    def reset_daily_tracking(self):
        """Reset daily P&L and trade tracking."""
        self.daily_pnl = Decimal('0')
        self.daily_trades = 0
        self.daily_start_balance = self.account_balance

        self.logger.info("Daily tracking reset")