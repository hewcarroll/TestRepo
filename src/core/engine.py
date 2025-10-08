"""
PDT Trading Bot Core Engine

Main trading engine that orchestrates strategy, risk management, data feeds,
and order execution in a multi-threaded environment.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import structlog
from threading import Lock, Event

from .strategy.momentum_strategy import MomentumStrategy
from .risk.risk_manager import RiskManager
from .data.market_feed import MarketDataFeed
from .execution.order_manager import OrderManager


class EngineState(Enum):
    """Trading engine operational states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TradingConfig:
    """Configuration for the trading engine."""
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    max_workers: int = 4
    data_buffer_size: int = 1000
    log_level: str = "INFO"
    enable_paper_trading: bool = True

    # PDT-specific settings
    min_trade_size: Decimal = Decimal('100')  # Minimum position size for PDT volume
    max_position_pct: Decimal = Decimal('0.05')  # 5% max position size
    daily_loss_limit_pct: Decimal = Decimal('0.01')  # 1% daily loss limit
    stop_loss_pct: Decimal = Decimal('0.02')  # 2% stop loss


@dataclass
class Position:
    """Current position information."""
    symbol: str
    quantity: int
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal('0')
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TradingEngine:
    """
    Multi-threaded trading engine for PDT optimization.

    Coordinates real-time market data, strategy execution, risk management,
    and order processing in a thread-safe manner.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.state = EngineState.STOPPED
        self._shutdown_event = Event()
        self._state_lock = Lock()

        # Initialize structured logging
        self.logger = structlog.get_logger("trading_engine")

        # Component instances (initialized after startup)
        self.strategy: Optional[MomentumStrategy] = None
        self.risk_manager: Optional[RiskManager] = None
        self.market_feed: Optional[MarketDataFeed] = None
        self.order_manager: Optional[OrderManager] = None

        # Thread-safe shared state
        self.positions: Dict[str, Position] = {}
        self.positions_lock = Lock()
        self.account_balance: Optional[Decimal] = None
        self.daily_pnl: Decimal = Decimal('0')
        self.trade_count: int = 0

        # Performance tracking
        self.start_time = None
        self.total_trades = 0
        self.successful_trades = 0
        self.total_volume = Decimal('0')

        # Thread pool for CPU-intensive tasks
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)

        self.logger.info("Trading engine initialized", max_workers=config.max_workers)

    async def start(self) -> bool:
        """
        Start the trading engine with all components.

        Returns:
            bool: True if startup successful, False otherwise
        """
        with self._state_lock:
            if self.state != EngineState.STOPPED:
                self.logger.warning("Cannot start engine in non-stopped state", state=self.state.value)
                return False

            self.state = EngineState.STARTING
            self.start_time = datetime.now(timezone.utc)

        try:
            self.logger.info("Starting trading engine components")

            # Initialize core components
            await self._initialize_components()

            # Start market data feed
            await self.market_feed.start()

            # Start strategy engine
            await self.strategy.initialize()

            # Start risk manager
            await self.risk_manager.initialize()

            # Start order manager
            await self.order_manager.initialize()

            with self._state_lock:
                self.state = EngineState.RUNNING

            self.logger.info("Trading engine started successfully")
            return True

        except Exception as e:
            self.logger.error("Failed to start trading engine", error=str(e), exc_info=True)
            with self._state_lock:
                self.state = EngineState.ERROR
            return False

    async def stop(self) -> bool:
        """
        Stop the trading engine and cleanup resources.

        Returns:
            bool: True if shutdown successful, False otherwise
        """
        with self._state_lock:
            if self.state not in [EngineState.RUNNING, EngineState.ERROR]:
                self.logger.warning("Cannot stop engine in current state", state=self.state.value)
                return False

            self.state = EngineState.STOPPING

        try:
            self.logger.info("Stopping trading engine")

            # Signal shutdown to all components
            self._shutdown_event.set()

            # Stop components in reverse order
            if self.order_manager:
                await self.order_manager.shutdown()
            if self.risk_manager:
                await self.risk_manager.shutdown()
            if self.strategy:
                await self.strategy.shutdown()
            if self.market_feed:
                await self.market_feed.stop()

            # Shutdown thread pool
            self._executor.shutdown(wait=True)

            with self._state_lock:
                self.state = EngineState.STOPPED

            # Log final statistics
            self._log_final_stats()

            self.logger.info("Trading engine stopped successfully")
            return True

        except Exception as e:
            self.logger.error("Error during engine shutdown", error=str(e), exc_info=True)
            return False

    async def _initialize_components(self):
        """Initialize all engine components."""
        # Initialize market data feed
        self.market_feed = MarketDataFeed(
            alpaca_api_key=self.config.alpaca_api_key,
            alpaca_secret_key=self.config.alpaca_secret_key,
            base_url=self.config.alpaca_base_url,
            buffer_size=self.config.data_buffer_size,
            shutdown_event=self._shutdown_event
        )

        # Initialize strategy
        self.strategy = MomentumStrategy(
            max_position_size=self.config.max_position_pct,
            shutdown_event=self._shutdown_event
        )

        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_position_pct=self.config.max_position_pct,
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            stop_loss_pct=self.config.stop_loss_pct,
            shutdown_event=self._shutdown_event
        )

        # Initialize order manager
        self.order_manager = OrderManager(
            alpaca_api_key=self.config.alpaca_api_key,
            alpaca_secret_key=self.config.alpaca_secret_key,
            base_url=self.config.alpaca_base_url,
            shutdown_event=self._shutdown_event
        )

        # Set up data flow callbacks
        self.market_feed.set_data_callback(self._on_market_data)
        self.strategy.set_signal_callback(self._on_trading_signal)
        self.risk_manager.set_risk_callback(self._on_risk_event)

    def _on_market_data(self, symbol: str, price_data: Dict[str, Any]):
        """Handle incoming market data."""
        try:
            # Update position prices if we hold this symbol
            with self.positions_lock:
                if symbol in self.positions:
                    position = self.positions[symbol]
                    old_price = position.current_price
                    position.current_price = Decimal(str(price_data['price']))
                    position.last_updated = datetime.now(timezone.utc)

                    # Calculate unrealized P&L
                    price_change = position.current_price - position.avg_entry_price
                    position.unrealized_pnl = price_change * Decimal(str(position.quantity))

                    self.logger.debug("Updated position price",
                                    symbol=symbol,
                                    old_price=float(old_price),
                                    new_price=float(position.current_price),
                                    unrealized_pnl=float(position.unrealized_pnl))

            # Forward to strategy for signal generation
            if self.strategy:
                asyncio.create_task(self.strategy.process_market_data(symbol, price_data))

        except Exception as e:
            self.logger.error("Error processing market data", symbol=symbol, error=str(e))

    def _on_trading_signal(self, signal: Dict[str, Any]):
        """Handle trading signals from strategy."""
        try:
            symbol = signal['symbol']
            action = signal['action']
            quantity = signal['quantity']
            price = signal['price']

            self.logger.info("Received trading signal",
                           symbol=symbol,
                           action=action,
                           quantity=quantity,
                           price=float(price))

            # Check risk limits before execution
            if self.risk_manager:
                risk_check = asyncio.create_task(
                    self.risk_manager.validate_trade(symbol, action, quantity, price)
                )

                # Execute trade if risk check passes
                if risk_check.result() and self.order_manager:
                    asyncio.create_task(
                        self.order_manager.execute_trade(symbol, action, quantity, price)
                    )

        except Exception as e:
            self.logger.error("Error processing trading signal", error=str(e), exc_info=True)

    def _on_risk_event(self, event: Dict[str, Any]):
        """Handle risk management events."""
        try:
            event_type = event['type']
            symbol = event.get('symbol')

            if event_type == 'stop_loss_triggered':
                self.logger.warning("Stop loss triggered", symbol=symbol, event=event)
                # Close position immediately
                if self.order_manager and symbol:
                    asyncio.create_task(
                        self.order_manager.close_position(symbol, "market", "close")
                    )

            elif event_type == 'daily_loss_limit':
                self.logger.error("Daily loss limit reached", event=event)
                # Shutdown engine to prevent further losses
                asyncio.create_task(self.emergency_stop())

            elif event_type == 'position_limit_exceeded':
                self.logger.warning("Position limit exceeded", symbol=symbol, event=event)

        except Exception as e:
            self.logger.error("Error processing risk event", error=str(e), exc_info=True)

    async def emergency_stop(self):
        """Emergency stop due to critical risk conditions."""
        self.logger.critical("Emergency stop initiated")
        await self.stop()

    def update_position(self, symbol: str, quantity: int, price: Decimal, action: str = "update"):
        """Thread-safe position update."""
        with self.positions_lock:
            if quantity == 0:
                # Position closed
                if symbol in self.positions:
                    position = self.positions.pop(symbol)
                    self.daily_pnl += position.unrealized_pnl + position.realized_pnl
                    self.logger.info("Position closed",
                                   symbol=symbol,
                                   total_pnl=float(position.unrealized_pnl + position.realized_pnl))
            else:
                # Update or create position
                if symbol in self.positions:
                    position = self.positions[symbol]
                    old_quantity = position.quantity

                    # Calculate realized P&L for partial closes
                    if abs(quantity) < abs(old_quantity) and action == "sell":
                        qty_change = old_quantity - quantity
                        price_change = price - position.avg_entry_price
                        realized_pnl = price_change * Decimal(str(qty_change))
                        position.realized_pnl += realized_pnl
                        self.daily_pnl += realized_pnl

                    position.quantity = quantity
                    position.current_price = price
                    position.last_updated = datetime.now(timezone.utc)

                    # Recalculate average entry price for new positions
                    if action == "buy" and old_quantity == 0:
                        position.avg_entry_price = price

                else:
                    # New position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_entry_price=price,
                        current_price=price
                    )

                self.trade_count += 1
                self.total_trades += 1

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        with self.positions_lock:
            total_value = Decimal('0')
            total_unrealized = Decimal('0')

            for position in self.positions.values():
                position_value = position.current_price * Decimal(str(abs(position.quantity)))
                total_value += position_value
                total_unrealized += position.unrealized_pnl

            return {
                'total_positions': len(self.positions),
                'total_value': float(total_value),
                'total_unrealized_pnl': float(total_unrealized),
                'daily_pnl': float(self.daily_pnl),
                'total_trades': self.total_trades,
                'success_rate': (self.successful_trades / max(self.total_trades, 1)) * 100,
                'account_balance': float(self.account_balance) if self.account_balance else 0,
                'positions': [
                    {
                        'symbol': p.symbol,
                        'quantity': p.quantity,
                        'avg_entry_price': float(p.avg_entry_price),
                        'current_price': float(p.current_price),
                        'unrealized_pnl': float(p.unrealized_pnl),
                        'realized_pnl': float(p.realized_pnl)
                    }
                    for p in self.positions.values()
                ]
            }

    def _log_final_stats(self):
        """Log final trading statistics."""
        uptime = datetime.now(timezone.utc) - self.start_time if self.start_time else None

        self.logger.info("Trading session completed",
                        uptime_seconds=uptime.total_seconds() if uptime else 0,
                        total_trades=self.total_trades,
                        successful_trades=self.successful_trades,
                        success_rate=(self.successful_trades / max(self.total_trades, 1)) * 100,
                        total_volume=float(self.total_volume),
                        final_pnl=float(self.daily_pnl),
                        final_positions=len(self.positions))

    def is_running(self) -> bool:
        """Check if engine is currently running."""
        with self._state_lock:
            return self.state == EngineState.RUNNING

    def get_state(self) -> EngineState:
        """Get current engine state."""
        with self._state_lock:
            return self.state