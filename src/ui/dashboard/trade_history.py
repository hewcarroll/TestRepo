"""
Trade History Module for PDT Trading Bot Admin UI

This module provides comprehensive trade history tracking, analysis,
and filtering capabilities for the trading dashboard.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TradeSide(Enum):
    """Trade side enumeration."""
    BUY = "buy"
    SELL = "sell"

class TradeStatus(Enum):
    """Trade status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    PENDING = "pending"

@dataclass
class TradeRecord:
    """Trade record structure."""
    trade_id: str
    symbol: str
    side: TradeSide
    quantity: float
    price: float
    timestamp: datetime
    pnl: float
    strategy: str
    status: TradeStatus
    commission: float = 0.0
    slippage: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rationale: str = ""

@dataclass
class TradeFilter:
    """Trade filtering criteria."""
    symbol: Optional[str] = None
    side: Optional[TradeSide] = None
    strategy: Optional[str] = None
    status: Optional[TradeStatus] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_pnl: Optional[float] = None
    max_pnl: Optional[float] = None
    min_quantity: Optional[float] = None
    max_quantity: Optional[float] = None

class TradeHistory:
    """Manages trade history and provides analysis capabilities."""

    def __init__(self, max_records: int = 10000):
        """Initialize trade history manager."""
        self.trades: List[TradeRecord] = []
        self.max_records = max_records
        self.trade_stats_cache: Dict[str, Any] = {}
        self.cache_timeout = 300  # 5 minutes
        self.last_cache_update = None

    async def add_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Add a new trade record.

        Args:
            trade_data: Trade data dictionary

        Returns:
            True if added successfully
        """
        try:
            trade = TradeRecord(
                trade_id=trade_data.get('trade_id', ''),
                symbol=trade_data.get('symbol', ''),
                side=TradeSide(trade_data.get('side', 'buy')),
                quantity=float(trade_data.get('quantity', 0)),
                price=float(trade_data.get('price', 0)),
                timestamp=datetime.fromisoformat(trade_data.get('timestamp', datetime.utcnow().isoformat())),
                pnl=float(trade_data.get('pnl', 0)),
                strategy=trade_data.get('strategy', ''),
                status=TradeStatus(trade_data.get('status', 'closed')),
                commission=float(trade_data.get('commission', 0)),
                slippage=float(trade_data.get('slippage', 0)),
                entry_time=datetime.fromisoformat(trade_data.get('entry_time')) if trade_data.get('entry_time') else None,
                exit_time=datetime.fromisoformat(trade_data.get('exit_time')) if trade_data.get('exit_time') else None,
                stop_loss=float(trade_data.get('stop_loss')) if trade_data.get('stop_loss') else None,
                take_profit=float(trade_data.get('take_profit')) if trade_data.get('take_profit') else None,
                rationale=trade_data.get('rationale', '')
            )

            self.trades.append(trade)

            # Sort by timestamp (newest first)
            self.trades.sort(key=lambda x: x.timestamp, reverse=True)

            # Maintain max records limit
            if len(self.trades) > self.max_records:
                self.trades = self.trades[:self.max_records]

            # Clear cache when new trade is added
            self.trade_stats_cache.clear()
            self.last_cache_update = None

            logger.info(f"Added trade: {trade.trade_id} - {trade.symbol} {trade.side.value} {trade.quantity} @ {trade.price}")
            return True

        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False

    async def get_trades(self, filter_criteria: Optional[TradeFilter] = None,
                        limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        Get filtered and paginated trades.

        Args:
            filter_criteria: Optional filtering criteria
            limit: Maximum number of trades to return
            offset: Offset for pagination

        Returns:
            Dictionary containing trades and metadata
        """
        try:
            # Apply filters
            filtered_trades = self.trades.copy()

            if filter_criteria:
                if filter_criteria.symbol:
                    filtered_trades = [t for t in filtered_trades if t.symbol == filter_criteria.symbol]

                if filter_criteria.side:
                    filtered_trades = [t for t in filtered_trades if t.side == filter_criteria.side]

                if filter_criteria.strategy:
                    filtered_trades = [t for t in filtered_trades if t.strategy == filter_criteria.strategy]

                if filter_criteria.status:
                    filtered_trades = [t for t in filtered_trades if t.status == filter_criteria.status]

                if filter_criteria.start_date:
                    filtered_trades = [t for t in filtered_trades if t.timestamp >= filter_criteria.start_date]

                if filter_criteria.end_date:
                    filtered_trades = [t for t in filtered_trades if t.timestamp <= filter_criteria.end_date]

                if filter_criteria.min_pnl is not None:
                    filtered_trades = [t for t in filtered_trades if t.pnl >= filter_criteria.min_pnl]

                if filter_criteria.max_pnl is not None:
                    filtered_trades = [t for t in filtered_trades if t.pnl <= filter_criteria.max_pnl]

                if filter_criteria.min_quantity is not None:
                    filtered_trades = [t for t in filtered_trades if t.quantity >= filter_criteria.min_quantity]

                if filter_criteria.max_quantity is not None:
                    filtered_trades = [t for t in filtered_trades if t.quantity <= filter_criteria.max_quantity]

            # Apply pagination
            total_count = len(filtered_trades)
            paginated_trades = filtered_trades[offset:offset + limit]

            # Convert to dictionaries for JSON serialization
            trade_dicts = []
            for trade in paginated_trades:
                trade_dict = {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'timestamp': trade.timestamp.isoformat(),
                    'pnl': trade.pnl,
                    'strategy': trade.strategy,
                    'status': trade.status.value,
                    'commission': trade.commission,
                    'slippage': trade.slippage,
                    'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit,
                    'rationale': trade.rationale
                }
                trade_dicts.append(trade_dict)

            return {
                'trades': trade_dicts,
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            }

        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return {
                'trades': [],
                'total_count': 0,
                'limit': limit,
                'offset': offset,
                'has_more': False
            }

    async def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive trade statistics.

        Returns:
            Trade statistics dictionary
        """
        try:
            # Check cache first
            if (self.trade_stats_cache and self.last_cache_update and
                (datetime.utcnow() - self.last_cache_update).seconds < self.cache_timeout):
                return self.trade_stats_cache

            if not self.trades:
                return self._get_empty_stats()

            # Basic statistics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = len([t for t in self.trades if t.pnl < 0])

            total_pnl = sum(t.pnl for t in self.trades)
            total_commission = sum(t.commission for t in self.trades)
            total_quantity = sum(t.quantity for t in self.trades)

            # Strategy breakdown
            strategy_stats = {}
            for trade in self.trades:
                if trade.strategy not in strategy_stats:
                    strategy_stats[trade.strategy] = {
                        'count': 0,
                        'total_pnl': 0,
                        'winning_trades': 0
                    }

                strategy_stats[trade.strategy]['count'] += 1
                strategy_stats[trade.strategy]['total_pnl'] += trade.pnl
                if trade.pnl > 0:
                    strategy_stats[trade.strategy]['winning_trades'] += 1

            # Symbol breakdown
            symbol_stats = {}
            for trade in self.trades:
                if trade.symbol not in symbol_stats:
                    symbol_stats[trade.symbol] = {
                        'count': 0,
                        'total_pnl': 0,
                        'total_quantity': 0
                    }

                symbol_stats[trade.symbol]['count'] += 1
                symbol_stats[trade.symbol]['total_pnl'] += trade.pnl
                symbol_stats[trade.symbol]['total_quantity'] += trade.quantity

            # Calculate averages
            avg_trade_size = total_quantity / total_trades if total_trades > 0 else 0
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

            # Calculate win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Calculate profit factor
            total_profits = sum(t.pnl for t in self.trades if t.pnl > 0)
            total_losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
            profit_factor = total_profits / total_losses if total_losses > 0 else 0

            # Largest win/loss
            largest_win = max((t.pnl for t in self.trades if t.pnl > 0), default=0)
            largest_loss = abs(min((t.pnl for t in self.trades if t.pnl < 0), default=0))

            # Cache results
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 4),
                'total_pnl': round(total_pnl, 2),
                'total_commission': round(total_commission, 2),
                'avg_trade_size': round(avg_trade_size, 2),
                'avg_pnl': round(avg_pnl, 2),
                'profit_factor': round(profit_factor, 4),
                'largest_win': round(largest_win, 2),
                'largest_loss': round(largest_loss, 2),
                'strategy_breakdown': strategy_stats,
                'symbol_breakdown': symbol_stats,
                'calculated_at': datetime.utcnow().isoformat()
            }

            self.trade_stats_cache = stats
            self.last_cache_update = datetime.utcnow()

            return stats

        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return self._get_empty_stats()

    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics structure."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_commission': 0.0,
            'avg_trade_size': 0.0,
            'avg_pnl': 0.0,
            'profit_factor': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'strategy_breakdown': {},
            'symbol_breakdown': {},
            'calculated_at': datetime.utcnow().isoformat()
        }

    async def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific trade by ID.

        Args:
            trade_id: Trade ID to look for

        Returns:
            Trade data dictionary or None if not found
        """
        try:
            trade = next((t for t in self.trades if t.trade_id == trade_id), None)

            if not trade:
                return None

            return {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'timestamp': trade.timestamp.isoformat(),
                'pnl': trade.pnl,
                'strategy': trade.strategy,
                'status': trade.status.value,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'rationale': trade.rationale
            }

        except Exception as e:
            logger.error(f"Error getting trade by ID: {e}")
            return None

    async def get_trades_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get trades for a specific symbol.

        Args:
            symbol: Symbol to filter by
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        try:
            symbol_trades = [t for t in self.trades if t.symbol == symbol][:limit]

            return [
                {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'timestamp': trade.timestamp.isoformat(),
                    'pnl': trade.pnl,
                    'strategy': trade.strategy,
                    'status': trade.status.value
                }
                for trade in symbol_trades
            ]

        except Exception as e:
            logger.error(f"Error getting trades by symbol: {e}")
            return []

    async def get_trades_by_strategy(self, strategy: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get trades for a specific strategy.

        Args:
            strategy: Strategy to filter by
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        try:
            strategy_trades = [t for t in self.trades if t.strategy == strategy][:limit]

            return [
                {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'timestamp': trade.timestamp.isoformat(),
                    'pnl': trade.pnl,
                    'strategy': trade.strategy,
                    'status': trade.status.value
                }
                for trade in strategy_trades
            ]

        except Exception as e:
            logger.error(f"Error getting trades by strategy: {e}")
            return []

    async def export_trade_history(self, format: str = 'json',
                                  filter_criteria: Optional[TradeFilter] = None) -> str:
        """
        Export trade history in specified format.

        Args:
            format: Export format ('json' or 'csv')
            filter_criteria: Optional filtering criteria

        Returns:
            Exported data as string
        """
        try:
            # Get filtered trades
            trades_data = await self.get_trades(filter_criteria, limit=10000)
            trades = trades_data['trades']

            if format.lower() == 'csv':
                # CSV export
                csv_lines = [
                    'Trade ID,Symbol,Side,Quantity,Price,Timestamp,P&L,Strategy,Status,Commission'
                ]

                for trade in trades:
                    csv_lines.append(
                        f"{trade['trade_id']},"
                        f"{trade['symbol']},"
                        f"{trade['side']},"
                        f"{trade['quantity']},"
                        f"{trade['price']},"
                        f"{trade['timestamp']},"
                        f"{trade['pnl']},"
                        f"{trade['strategy']},"
                        f"{trade['status']},"
                        f"{trade['commission']}"
                    )

                return '\n'.join(csv_lines)

            else:
                # JSON export
                export_data = {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'total_trades': len(trades),
                    'trades': trades
                }

                return json.dumps(export_data, indent=2)

        except Exception as e:
            logger.error(f"Error exporting trade history: {e}")
            return ""

    async def clear_trade_history(self) -> bool:
        """
        Clear all trade history.

        Returns:
            True if cleared successfully
        """
        try:
            self.trades.clear()
            self.trade_stats_cache.clear()
            self.last_cache_update = None
            logger.info("Trade history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing trade history: {e}")
            return False

    async def get_trade_performance_by_period(self, days: int = 30) -> Dict[str, Any]:
        """
        Get trade performance metrics for a specific period.

        Args:
            days: Number of days to analyze

        Returns:
            Performance metrics for the period
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            period_trades = [t for t in self.trades if t.timestamp >= cutoff_date]

            if not period_trades:
                return {
                    'period_days': days,
                    'total_trades': 0,
                    'performance': {}
                }

            # Calculate period-specific metrics
            total_pnl = sum(t.pnl for t in period_trades)
            winning_trades = len([t for t in period_trades if t.pnl > 0])
            total_quantity = sum(t.quantity for t in period_trades)

            return {
                'period_days': days,
                'total_trades': len(period_trades),
                'winning_trades': winning_trades,
                'losing_trades': len(period_trades) - winning_trades,
                'win_rate': round(winning_trades / len(period_trades), 4) if period_trades else 0,
                'total_pnl': round(total_pnl, 2),
                'total_quantity': round(total_quantity, 2),
                'avg_trade_size': round(total_quantity / len(period_trades), 2) if period_trades else 0,
                'start_date': cutoff_date.isoformat(),
                'end_date': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating period performance: {e}")
            return {
                'period_days': days,
                'total_trades': 0,
                'performance': {}
            }

# Global trade history instance
trade_history = TradeHistory()
