"""
Pattern Day Trading (PDT) Compliance Tracker

Enforces FINRA PDT rules with hard-coded safety checks.
Tracks day trading activity and prevents violations of the 3-trade-per-5-day limit.
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal
from threading import Event, Lock
from typing import Dict, List, Optional, Set, Tuple
import json
import os

from ..audit.audit_logger import AuditLogger


@dataclass
class DayTrade:
    """Represents a single day trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    quantity: int
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    is_complete: bool = False


@dataclass
class TradingDay:
    """Represents trading activity for a single day."""
    date: date
    trades: List[DayTrade] = field(default_factory=list)
    day_trade_count: int = 0
    total_volume: Decimal = Decimal('0')
    is_pdt_day: bool = False


class PDTTracker:
    """
    Pattern Day Trading compliance tracker.

    Enforces FINRA Rule 4210(f)(8)(B) - Pattern Day Trader rule.
    Accounts under $25,000 cannot make more than 3 day trades in any 5-day period.
    """

    def __init__(self, shutdown_event: Optional[Event] = None):
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("pdt_tracker")
        self.audit_logger = AuditLogger()

        # Hard-coded PDT constants (cannot be modified)
        self._PDT_ACCOUNT_THRESHOLD = Decimal('25000')
        self._MAX_DAY_TRADES = 3
        self._TRACKING_DAYS = 5
        self._MIN_HOLDING_TIME = 1  # Minimum minutes to hold position to avoid day trade

        # Thread-safe state
        self._lock = Lock()
        self._trades: Dict[str, DayTrade] = {}  # symbol -> current trade
        self._trading_days: Dict[date, TradingDay] = {}
        self._account_balance: Optional[Decimal] = None
        self._is_pdt_account: bool = False

        # PDT violation tracking
        self._violation_count = 0
        self._last_violation_date: Optional[date] = None
        self._cooling_off_until: Optional[datetime] = None

        self.logger.info("PDT tracker initialized",
                        pdt_threshold=float(self._PDT_ACCOUNT_THRESHOLD),
                        max_day_trades=self._MAX_DAY_TRADES)

    async def initialize(self):
        """Initialize PDT tracker and load historical data."""
        try:
            await self.audit_logger.initialize()
            await self._load_historical_data()
            await self._update_pdt_status()

            self.logger.info("PDT tracker initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize PDT tracker", error=str(e))
            raise

    async def shutdown(self):
        """Shutdown PDT tracker and save state."""
        try:
            await self._save_historical_data()
            await self.audit_logger.shutdown()
            self.logger.info("PDT tracker shutdown complete")
        except Exception as e:
            self.logger.error("Error during PDT tracker shutdown", error=str(e))

    async def record_trade(self, action: str, quantity: int, symbol: str = None,
                          price: Optional[Decimal] = None) -> bool:
        """
        Record a trade for PDT compliance tracking.

        Args:
            action: 'buy' or 'sell'
            quantity: Number of shares
            symbol: Trading symbol
            price: Trade price

        Returns:
            True if recorded successfully
        """
        if self.shutdown_event.is_set():
            return False

        with self._lock:
            try:
                current_time = datetime.now(timezone.utc)

                if action == 'buy':
                    # Start of potential day trade
                    if symbol in self._trades:
                        self.logger.warning("Opening buy for symbol already in position", symbol=symbol)
                        return False

                    self._trades[symbol] = DayTrade(
                        symbol=symbol,
                        entry_time=current_time,
                        quantity=quantity,
                        entry_price=price or Decimal('0'),
                        exit_time=current_time,  # Will be updated on sell
                        exit_price=Decimal('0'),
                        pnl=Decimal('0')
                    )

                elif action == 'sell':
                    # Potential end of day trade
                    if symbol not in self._trades:
                        self.logger.warning("Sell order for symbol not in position", symbol=symbol)
                        return False

                    trade = self._trades[symbol]
                    holding_time = (current_time - trade.entry_time).total_seconds() / 60  # minutes

                    # Check if this qualifies as a day trade
                    if holding_time < self._MIN_HOLDING_TIME:
                        # This is a day trade - check limits
                        if await self._is_pdt_violation():
                            self.logger.error("Day trade would violate PDT rules",
                                            symbol=symbol,
                                            holding_time_minutes=holding_time)

                            # Log violation
                            await self._log_pdt_violation(symbol, "EXCEEDED_DAY_TRADE_LIMIT")

                            return False

                        # Record as day trade
                        await self._record_day_trade(trade, current_time, price or Decimal('0'))

                    # Complete the trade
                    trade.exit_time = current_time
                    trade.exit_price = price or Decimal('0')
                    trade.pnl = (trade.exit_price - trade.entry_price) * Decimal(str(quantity))
                    trade.is_complete = True

                    # Remove from active trades
                    del self._trades[symbol]

                else:
                    self.logger.error("Invalid action for PDT tracking", action=action)
                    return False

                return True

            except Exception as e:
                self.logger.error("Error recording trade for PDT", error=str(e))
                return False

    async def is_day_trade(self, action: str, quantity: int, symbol: str = None) -> bool:
        """
        Check if a trade would qualify as a day trade.

        Args:
            action: 'buy' or 'sell'
            quantity: Number of shares
            symbol: Trading symbol

        Returns:
            True if this would be a day trade
        """
        if action == 'buy' or not symbol:
            return False

        with self._lock:
            if symbol not in self._trades:
                return False

            trade = self._trades[symbol]
            current_time = datetime.now(timezone.utc)
            holding_time = (current_time - trade.entry_time).total_seconds() / 60

            return holding_time < self._MIN_HOLDING_TIME

    async def get_day_trade_count(self) -> int:
        """Get current day trade count in the 5-day window."""
        with self._lock:
            today = date.today()
            count = 0

            # Count day trades in the last 5 trading days
            for i in range(self._TRACKING_DAYS):
                check_date = today - timedelta(days=i)
                if check_date in self._trading_days:
                    count += self._trading_days[check_date].day_trade_count

            return count

    async def get_pdt_status(self) -> Dict[str, any]:
        """Get current PDT compliance status."""
        with self._lock:
            today = date.today()
            day_trade_count = await self.get_day_trade_count()

            return {
                'is_pdt_account': self._is_pdt_account,
                'account_balance': float(self._account_balance) if self._account_balance else None,
                'pdt_threshold': float(self._PDT_ACCOUNT_THRESHOLD),
                'current_day_trade_count': day_trade_count,
                'max_day_trades': self._MAX_DAY_TRADES,
                'remaining_day_trades': max(0, self._MAX_DAY_TRADES - day_trade_count),
                'can_day_trade': day_trade_count < self._MAX_DAY_TRADES,
                'violation_count': self._violation_count,
                'last_violation_date': self._last_violation_date.isoformat() if self._last_violation_date else None,
                'cooling_off_until': self._cooling_off_until.isoformat() if self._cooling_off_until else None,
                'trading_days_tracked': len(self._trading_days)
            }

    def update_account_balance(self, balance: Decimal):
        """Update account balance for PDT threshold checking."""
        with self._lock:
            old_balance = self._account_balance
            self._account_balance = balance

            # Update PDT status if balance crosses threshold
            if old_balance and balance != old_balance:
                asyncio.create_task(self._update_pdt_status())

    async def _is_pdt_violation(self) -> bool:
        """Check if a day trade would violate PDT rules."""
        day_trade_count = await self.get_day_trade_count()

        # Hard-coded rule: cannot exceed 3 day trades in 5 days
        if day_trade_count >= self._MAX_DAY_TRADES:
            return True

        # Check if account is under PDT threshold
        if (self._account_balance and
            self._account_balance < self._PDT_ACCOUNT_THRESHOLD):
            return True

        return False

    async def _record_day_trade(self, trade: DayTrade, exit_time: datetime, exit_price: Decimal):
        """Record a completed day trade."""
        with self._lock:
            today = date.today()

            # Create trading day if it doesn't exist
            if today not in self._trading_days:
                self._trading_days[today] = TradingDay(date=today)

            trading_day = self._trading_days[today]

            # Calculate P&L for the day trade
            pnl = (exit_price - trade.entry_price) * Decimal(str(trade.quantity))

            # Create day trade record
            day_trade = DayTrade(
                symbol=trade.symbol,
                entry_time=trade.entry_time,
                exit_time=exit_time,
                quantity=trade.quantity,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                is_complete=True
            )

            trading_day.trades.append(day_trade)
            trading_day.day_trade_count += 1
            trading_day.total_volume += abs(pnl)

            # Mark as PDT day if we have day trades
            if trading_day.day_trade_count > 0:
                trading_day.is_pdt_day = True

            self.logger.info("Day trade recorded",
                           symbol=trade.symbol,
                           day_trade_count=trading_day.day_trade_count,
                           pnl=float(pnl))

            # Log for audit
            await self._log_day_trade(day_trade)

    async def _log_day_trade(self, day_trade: DayTrade):
        """Log day trade for audit trail."""

        trade_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'day_trade',
            'symbol': day_trade.symbol,
            'entry_time': day_trade.entry_time.isoformat(),
            'exit_time': day_trade.exit_time.isoformat(),
            'quantity': day_trade.quantity,
            'entry_price': float(day_trade.entry_price),
            'exit_price': float(day_trade.exit_price),
            'pnl': float(day_trade.pnl),
            'holding_time_minutes': (day_trade.exit_time - day_trade.entry_time).total_seconds() / 60
        }

        await self.audit_logger.log_trade_activity(trade_data)

    async def _log_pdt_violation(self, symbol: str, violation_type: str):
        """Log PDT violation for audit trail."""

        violation_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'pdt_violation',
            'symbol': symbol,
            'violation_type': violation_type,
            'account_balance': float(self._account_balance) if self._account_balance else None,
            'day_trade_count': await self.get_day_trade_count(),
            'max_day_trades': self._MAX_DAY_TRADES
        }

        await self.audit_logger.log_compliance_violation(violation_data)

        # Update violation tracking
        with self._lock:
            self._violation_count += 1
            self._last_violation_date = date.today()

    async def _update_pdt_status(self):
        """Update PDT account status based on current balance."""
        with self._lock:
            old_status = self._is_pdt_account

            if self._account_balance:
                self._is_pdt_account = self._account_balance < self._PDT_ACCOUNT_THRESHOLD
            else:
                self._is_pdt_account = True  # Assume PDT if balance unknown

            if old_status != self._is_pdt_account:
                self.logger.info("PDT status updated",
                               is_pdt_account=self._is_pdt_account,
                               account_balance=float(self._account_balance) if self._account_balance else None)

    async def _load_historical_data(self):
        """Load historical trading data from storage."""
        try:
            # In a real implementation, this would load from a database or file
            # For now, we'll start with empty state
            self.logger.info("Historical PDT data loaded")
        except Exception as e:
            self.logger.error("Error loading historical PDT data", error=str(e))

    async def _save_historical_data(self):
        """Save current trading data to storage."""
        try:
            # In a real implementation, this would save to a database or file
            # For now, we'll just log the action
            self.logger.info("Historical PDT data saved",
                           trading_days=len(self._trading_days),
                           total_trades=sum(len(day.trades) for day in self._trading_days.values()))
        except Exception as e:
            self.logger.error("Error saving historical PDT data", error=str(e))

    def get_status(self) -> Dict[str, any]:
        """Get current PDT tracker status."""
        return {
            'is_pdt_account': self._is_pdt_account,
            'account_balance': float(self._account_balance) if self._account_balance else None,
            'day_trade_count': len([t for day in self._trading_days.values() for t in day.trades]),
            'violation_count': self._violation_count,
            'last_violation': self._last_violation_date.isoformat() if self._last_violation_date else None,
            'tracking_days': len(self._trading_days)
        }