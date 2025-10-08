"""
Order Management Module for PDT Trading Bot

Handles order placement, execution, position tracking, and P&L calculations
with Alpaca brokerage integration.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from threading import Event
from typing import Dict, List, Optional, Any, Callable
import structlog
import aiohttp


class OrderStatus(Enum):
    """Order status values."""
    PENDING = "pending"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type values."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side values."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Trading order representation."""
    id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: int
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "day"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: Optional[Decimal] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    alpaca_order_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class Position:
    """Current position information."""
    symbol: str
    quantity: int
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal('0')
    market_value: Decimal = Decimal('0')
    cost_basis: Decimal = Decimal('0')
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExecutionConfig:
    """Order execution configuration."""
    alpaca_api_key: str
    alpaca_secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"
    max_retries: int = 3
    retry_delay: float = 1.0
    order_timeout: int = 30  # seconds
    position_update_interval: int = 5  # seconds


class OrderManager:
    """
    Order management and execution system for PDT trading.

    Handles order placement, execution tracking, position management,
    and P&L calculations with Alpaca integration.
    """

    def __init__(self,
                 alpaca_api_key: str,
                 alpaca_secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets",
                 shutdown_event: Optional[Event] = None):
        self.config = ExecutionConfig(
            alpaca_api_key=alpaca_api_key,
            alpaca_secret_key=alpaca_secret_key,
            base_url=base_url
        )
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("order_manager")

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.account_info: Optional[Dict[str, Any]] = None

        # Session tracking
        self.session_orders: int = 0
        self.session_filled_orders: int = 0
        self.session_volume: Decimal = Decimal('0')
        self.session_pnl: Decimal = Decimal('0')

        # Performance metrics
        self.total_orders = 0
        self.total_filled_orders = 0
        self.total_volume = Decimal('0')
        self.total_pnl = Decimal('0')

        # HTTP session for API calls
        self._session: Optional[aiohttp.ClientSession] = None

        self.logger.info("Order manager initialized", base_url=base_url)

    async def initialize(self):
        """Initialize order manager and establish API connection."""
        self.logger.info("Initializing order manager")

        try:
            # Create HTTP session
            headers = {
                'APCA-API-KEY-ID': self.config.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.config.alpaca_secret_key
            }

            self._session = aiohttp.ClientSession(headers=headers)

            # Fetch initial account information
            await self._update_account_info()

            # Start position monitoring
            asyncio.create_task(self._position_monitor_loop())

            self.logger.info("Order manager initialized successfully",
                           account_id=self.account_info.get('id') if self.account_info else None)

        except Exception as e:
            self.logger.error("Failed to initialize order manager", error=str(e))
            raise

    async def shutdown(self):
        """Shutdown order manager and cleanup resources."""
        self.logger.info("Shutting down order manager")

        try:
            # Cancel all pending orders
            await self._cancel_all_orders()

            # Close HTTP session
            if self._session:
                await self._session.close()
                self._session = None

            # Clear tracking data
            self.orders.clear()
            self.positions.clear()

            self.logger.info("Order manager shutdown complete")

        except Exception as e:
            self.logger.error("Error during order manager shutdown", error=str(e))

    async def execute_trade(self, symbol: str, action: str, quantity: int, price: Optional[float] = None) -> Optional[str]:
        """
        Execute a trade order.

        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Limit price (optional, for market orders)

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if self.shutdown_event.is_set():
                return None

            # Validate order parameters
            if not self._validate_order(symbol, action, quantity, price):
                return None

            # Create order object
            order_id = str(uuid.uuid4())
            order = Order(
                id=order_id,
                symbol=symbol,
                order_type=OrderType.LIMIT if price else OrderType.MARKET,
                side=OrderSide.BUY if action == "buy" else OrderSide.SELL,
                quantity=quantity,
                price=Decimal(str(price)) if price else None
            )

            self.orders[order_id] = order
            self.session_orders += 1
            self.total_orders += 1

            # Submit to Alpaca
            success = await self._submit_order(order)

            if success:
                self.logger.info("Order submitted successfully",
                               order_id=order_id,
                               symbol=symbol,
                               action=action,
                               quantity=quantity,
                               price=price)
                return order_id
            else:
                # Remove failed order
                if order_id in self.orders:
                    del self.orders[order_id]
                return None

        except Exception as e:
            self.logger.error("Error executing trade",
                            symbol=symbol,
                            action=action,
                            error=str(e))
            return None

    async def close_position(self, symbol: str, order_type: str = "market", time_in_force: str = "day") -> Optional[str]:
        """
        Close entire position for symbol.

        Args:
            symbol: Trading symbol
            order_type: Order type ('market' or 'limit')
            time_in_force: Time in force for limit orders

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Get current position
            if symbol not in self.positions:
                self.logger.warning("No position to close", symbol=symbol)
                return None

            position = self.positions[symbol]
            quantity = abs(position.quantity)

            if quantity == 0:
                return None

            # Determine side based on current position
            side = "sell" if position.quantity > 0 else "buy"

            # Use current price for limit orders
            price = None
            if order_type == "limit":
                price = float(position.current_price)

            return await self.execute_trade(symbol, side, quantity, price)

        except Exception as e:
            self.logger.error("Error closing position",
                            symbol=symbol,
                            error=str(e))
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            if order_id not in self.orders:
                self.logger.warning("Order not found for cancellation", order_id=order_id)
                return False

            order = self.orders[order_id]

            if order.status not in [OrderStatus.PENDING, OrderStatus.NEW]:
                self.logger.warning("Cannot cancel order in current status",
                                  order_id=order_id,
                                  status=order.status.value)
                return False

            # Cancel with Alpaca
            success = await self._cancel_alpaca_order(order)

            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now(timezone.utc)
                self.logger.info("Order cancelled", order_id=order_id)
                return True

            return False

        except Exception as e:
            self.logger.error("Error cancelling order",
                            order_id=order_id,
                            error=str(e))
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get current status of an order."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return None

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self.positions.copy()

    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary with P&L information."""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_market_value = sum(pos.market_value for pos in self.positions.values())

        return {
            'account_balance': float(self.account_info.get('cash', '0')) if self.account_info else 0,
            'total_market_value': float(total_market_value),
            'total_unrealized_pnl': float(total_unrealized_pnl),
            'total_realized_pnl': float(total_realized_pnl),
            'total_pnl': float(total_unrealized_pnl + total_realized_pnl),
            'num_positions': len(self.positions),
            'session_orders': self.session_orders,
            'session_filled_orders': self.session_filled_orders,
            'session_volume': float(self.session_volume),
            'session_pnl': float(self.session_pnl),
            'total_orders': self.total_orders,
            'total_filled_orders': self.total_filled_orders,
            'total_volume': float(self.total_volume),
            'total_pnl': float(self.total_pnl)
        }

    def _validate_order(self, symbol: str, action: str, quantity: int, price: Optional[float] = None) -> bool:
        """Validate order parameters."""
        if quantity <= 0:
            self.logger.warning("Invalid quantity", quantity=quantity)
            return False

        if action not in ["buy", "sell"]:
            self.logger.warning("Invalid action", action=action)
            return False

        if price is not None and price <= 0:
            self.logger.warning("Invalid price", price=price)
            return False

        return True

    async def _submit_order(self, order: Order) -> bool:
        """Submit order to Alpaca."""
        if not self._session:
            self.logger.error("HTTP session not initialized")
            return False

        try:
            # Prepare order data
            order_data = {
                'symbol': order.symbol,
                'qty': str(order.quantity),
                'side': order.side.value,
                'type': order.order_type.value,
                'time_in_force': order.time_in_force
            }

            if order.price:
                order_data['limit_price'] = str(order.price)

            # Submit order
            url = f"{self.config.base_url}/v2/orders"

            async with self._session.post(url, json=order_data) as response:
                if response.status == 201:
                    alpaca_order = await response.json()
                    order.alpaca_order_id = alpaca_order['id']
                    order.status = OrderStatus.NEW
                    order.updated_at = datetime.now(timezone.utc)

                    # Start order monitoring
                    asyncio.create_task(self._monitor_order(order.id))

                    return True
                else:
                    error_text = await response.text()
                    order.status = OrderStatus.REJECTED
                    order.error_message = f"HTTP {response.status}: {error_text}"
                    order.updated_at = datetime.now(timezone.utc)

                    self.logger.error("Order rejected",
                                    order_id=order.id,
                                    status=response.status,
                                    error=error_text)
                    return False

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            order.updated_at = datetime.now(timezone.utc)

            self.logger.error("Error submitting order",
                            order_id=order.id,
                            error=str(e))
            return False

    async def _cancel_alpaca_order(self, order: Order) -> bool:
        """Cancel order with Alpaca."""
        if not self._session or not order.alpaca_order_id:
            return False

        try:
            url = f"{self.config.base_url}/v2/orders/{order.alpaca_order_id}"

            async with self._session.delete(url) as response:
                return response.status == 204

        except Exception as e:
            self.logger.error("Error cancelling Alpaca order",
                            order_id=order.id,
                            alpaca_id=order.alpaca_order_id,
                            error=str(e))
            return False

    async def _cancel_all_orders(self):
        """Cancel all pending orders."""
        pending_orders = [
            order_id for order_id, order in self.orders.items()
            if order.status in [OrderStatus.PENDING, OrderStatus.NEW]
        ]

        for order_id in pending_orders:
            await self.cancel_order(order_id)

    async def _monitor_order(self, order_id: str):
        """Monitor order status until completion."""
        order = self.orders.get(order_id)
        if not order:
            return

        start_time = datetime.now(timezone.utc)
        timeout = timedelta(seconds=self.config.order_timeout)

        while (datetime.now(timezone.utc) - start_time) < timeout:
            try:
                if self.shutdown_event.is_set():
                    break

                # Check order status with Alpaca
                await self._check_order_status(order)

                # Exit if order is final status
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED,
                                  OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    break

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                self.logger.error("Error monitoring order",
                                order_id=order_id,
                                error=str(e))
                break

        # Handle timeout
        if order.status in [OrderStatus.PENDING, OrderStatus.NEW]:
            self.logger.warning("Order monitoring timeout", order_id=order_id)
            order.status = OrderStatus.EXPIRED

    async def _check_order_status(self, order: Order):
        """Check order status with Alpaca."""
        if not self._session or not order.alpaca_order_id:
            return

        try:
            url = f"{self.config.base_url}/v2/orders/{order.alpaca_order_id}"

            async with self._session.get(url) as response:
                if response.status == 200:
                    alpaca_order = await response.json()

                    # Update order status
                    old_status = order.status
                    order.status = OrderStatus(alpaca_order['status'])
                    order.filled_quantity = int(alpaca_order.get('filled_qty', '0'))

                    if alpaca_order.get('filled_avg_price'):
                        order.avg_fill_price = Decimal(alpaca_order['filled_avg_price'])

                    order.updated_at = datetime.now(timezone.utc)

                    # Handle filled order
                    if (old_status != OrderStatus.FILLED and
                        order.status == OrderStatus.FILLED):

                        await self._handle_filled_order(order)

                elif response.status == 404:
                    # Order not found
                    order.status = OrderStatus.REJECTED
                    order.error_message = "Order not found"

        except Exception as e:
            self.logger.error("Error checking order status",
                            order_id=order.id,
                            error=str(e))

    async def _handle_filled_order(self, order: Order):
        """Handle order fill event."""
        try:
            # Update session statistics
            self.session_filled_orders += 1
            self.total_filled_orders += 1

            fill_value = Decimal(str(order.filled_quantity)) * (order.avg_fill_price or order.price or Decimal('0'))
            self.session_volume += fill_value
            self.total_volume += fill_value

            # Update position
            await self._update_position(order)

            self.logger.info("Order filled",
                           order_id=order.id,
                           symbol=order.symbol,
                           side=order.side.value,
                           quantity=order.filled_quantity,
                           avg_fill_price=float(order.avg_fill_price) if order.avg_fill_price else None)

        except Exception as e:
            self.logger.error("Error handling filled order",
                            order_id=order.id,
                            error=str(e))

    async def _update_position(self, order: Order):
        """Update position after order fill."""
        try:
            symbol = order.symbol
            quantity = order.filled_quantity
            if order.side == OrderSide.SELL:
                quantity = -quantity

            fill_price = order.avg_fill_price or order.price or Decimal('0')

            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                old_quantity = position.quantity

                # Calculate new average entry price
                if old_quantity * quantity > 0:  # Same direction
                    total_quantity = old_quantity + quantity
                    if total_quantity != 0:
                        position.avg_entry_price = (
                            (position.avg_entry_price * Decimal(str(old_quantity)) +
                             fill_price * Decimal(str(quantity))) / Decimal(str(total_quantity))
                        )
                else:  # Opposite direction (partial close)
                    if abs(quantity) < abs(old_quantity):
                        # Partial close - keep existing entry price
                        pass
                    else:
                        # Full close or reversal
                        position.avg_entry_price = fill_price

                position.quantity += quantity
                position.cost_basis = position.avg_entry_price * Decimal(str(abs(position.quantity)))

            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=Decimal('0'),
                    cost_basis=fill_price * Decimal(str(abs(quantity)))
                )

            # Update market value and P&L
            await self._update_position_pnl(symbol)

        except Exception as e:
            self.logger.error("Error updating position",
                            order_id=order.id,
                            symbol=order.symbol,
                            error=str(e))

    async def _update_position_pnl(self, symbol: str):
        """Update position P&L with current market price."""
        if symbol not in self.positions:
            return

        try:
            # Get current market price from Alpaca
            current_price = await self._get_current_price(symbol)

            if current_price:
                position = self.positions[symbol]
                position.current_price = current_price
                position.market_value = current_price * Decimal(str(abs(position.quantity)))

                # Calculate unrealized P&L
                price_change = current_price - position.avg_entry_price
                position.unrealized_pnl = price_change * Decimal(str(position.quantity))

                position.last_updated = datetime.now(timezone.utc)

        except Exception as e:
            self.logger.error("Error updating position P&L",
                            symbol=symbol,
                            error=str(e))

    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for symbol."""
        if not self._session:
            return None

        try:
            url = f"{self.config.base_url}/v2/stocks/{symbol}/quotes/latest"

            async with self._session.get(url) as response:
                if response.status == 200:
                    quote = await response.json()
                    if quote.get('quote') and quote['quote'].get('ask_price'):
                        return Decimal(str(quote['quote']['ask_price']))

            return None

        except Exception as e:
            self.logger.error("Error getting current price",
                            symbol=symbol,
                            error=str(e))
            return None

    async def _update_account_info(self):
        """Update account information from Alpaca."""
        if not self._session:
            return

        try:
            url = f"{self.config.base_url}/v2/account"

            async with self._session.get(url) as response:
                if response.status == 200:
                    self.account_info = await response.json()

        except Exception as e:
            self.logger.error("Error updating account info", error=str(e))

    async def _position_monitor_loop(self):
        """Monitor and update all positions periodically."""
        while not self.shutdown_event.is_set():
            try:
                # Update all positions
                for symbol in list(self.positions.keys()):
                    await self._update_position_pnl(symbol)

                # Update account info periodically
                await self._update_account_info()

                await asyncio.sleep(self.config.position_update_interval)

            except Exception as e:
                self.logger.error("Error in position monitor loop", error=str(e))
                await asyncio.sleep(self.config.position_update_interval)