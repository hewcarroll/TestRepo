"""
P&L Monitor Module for PDT Trading Bot Admin UI

This module provides comprehensive P&L tracking, performance analysis,
and visualization for the trading dashboard.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import logging

logger = logging.getLogger(__name__)

@dataclass
class PnLData:
    """P&L data structure."""
    timestamp: datetime
    daily_pnl: float
    total_pnl: float
    daily_return: float
    total_return: float
    realized_pnl: float
    unrealized_pnl: float
    commission: float
    slippage: float

@dataclass
class PerformanceMetrics:
    """Performance metrics structure."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    calmar_ratio: float

class PnLMonitor:
    """Monitors and analyzes P&L data for the trading bot."""

    def __init__(self):
        """Initialize P&L monitor."""
        self.pnl_history: List[PnLData] = []
        self.performance_cache: Dict[str, Any] = {}
        self.cache_timeout = 300  # 5 minutes
        self.last_cache_update = None

        # Risk-free rate for Sharpe ratio calculation (2% annual)
        self.risk_free_rate = 0.02

    async def record_pnl_update(self, pnl_data: Dict[str, Any]) -> None:
        """
        Record a P&L update.

        Args:
            pnl_data: P&L data dictionary
        """
        try:
            pnl_record = PnLData(
                timestamp=datetime.utcnow(),
                daily_pnl=pnl_data.get('daily_pnl', 0.0),
                total_pnl=pnl_data.get('total_pnl', 0.0),
                daily_return=pnl_data.get('daily_return', 0.0),
                total_return=pnl_data.get('total_return', 0.0),
                realized_pnl=pnl_data.get('realized_pnl', 0.0),
                unrealized_pnl=pnl_data.get('unrealized_pnl', 0.0),
                commission=pnl_data.get('commission', 0.0),
                slippage=pnl_data.get('slippage', 0.0)
            )

            self.pnl_history.append(pnl_record)

            # Keep only last 30 days of data
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            self.pnl_history = [
                record for record in self.pnl_history
                if record.timestamp > cutoff_date
            ]

            # Clear performance cache when new data arrives
            self.performance_cache.clear()
            self.last_cache_update = None

            logger.info(f"Recorded P&L update: Daily=${pnl_record.daily_pnl}, Total=${pnl_record.total_pnl}")

        except Exception as e:
            logger.error(f"Error recording P&L update: {e}")

    async def get_current_pnl(self) -> Dict[str, Any]:
        """
        Get current P&L data.

        Returns:
            Current P&L information
        """
        try:
            if not self.pnl_history:
                return {
                    'daily_pnl': 0.0,
                    'total_pnl': 0.0,
                    'daily_return': 0.0,
                    'total_return': 0.0,
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0,
                    'commission': 0.0,
                    'slippage': 0.0,
                    'timestamp': datetime.utcnow().isoformat()
                }

            latest = self.pnl_history[-1]

            return {
                'daily_pnl': round(latest.daily_pnl, 2),
                'total_pnl': round(latest.total_pnl, 2),
                'daily_return': round(latest.daily_return, 4),
                'total_return': round(latest.total_return, 4),
                'realized_pnl': round(latest.realized_pnl, 2),
                'unrealized_pnl': round(latest.unrealized_pnl, 2),
                'commission': round(latest.commission, 2),
                'slippage': round(latest.slippage, 2),
                'timestamp': latest.timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting current P&L: {e}")
            return {}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Performance metrics dictionary
        """
        try:
            # Check cache first
            if (self.performance_cache and self.last_cache_update and
                (datetime.utcnow() - self.last_cache_update).seconds < self.cache_timeout):
                return self.performance_cache

            if len(self.pnl_history) < 2:
                return self._get_empty_metrics()

            # Extract daily returns for calculations
            daily_returns = []
            daily_pnls = []

            for record in self.pnl_history:
                if record.daily_return != 0:
                    daily_returns.append(record.daily_return)
                if record.daily_pnl != 0:
                    daily_pnls.append(record.daily_pnl)

            if not daily_returns:
                return self._get_empty_metrics()

            # Calculate metrics
            metrics = PerformanceMetrics(
                sharpe_ratio=self._calculate_sharpe_ratio(daily_returns),
                sortino_ratio=self._calculate_sortino_ratio(daily_returns),
                max_drawdown=self._calculate_max_drawdown(),
                win_rate=self._calculate_win_rate(daily_pnls),
                avg_win=self._calculate_avg_win(daily_pnls),
                avg_loss=self._calculate_avg_loss(daily_pnls),
                profit_factor=self._calculate_profit_factor(daily_pnls),
                calmar_ratio=self._calculate_calmar_ratio(daily_returns)
            )

            # Cache results
            result = {
                'sharpe_ratio': round(metrics.sharpe_ratio, 4),
                'sortino_ratio': round(metrics.sortino_ratio, 4),
                'max_drawdown': round(metrics.max_drawdown, 4),
                'win_rate': round(metrics.win_rate, 4),
                'avg_win': round(metrics.avg_win, 2),
                'avg_loss': round(metrics.avg_loss, 2),
                'profit_factor': round(metrics.profit_factor, 4),
                'calmar_ratio': round(metrics.calmar_ratio, 4),
                'calculated_at': datetime.utcnow().isoformat()
            }

            self.performance_cache = result
            self.last_cache_update = datetime.utcnow()

            return result

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._get_empty_metrics()

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0

        if std_return == 0:
            return 0.0

        # Annualize the ratio (assuming daily returns)
        annual_return = avg_return * 252
        annual_std = std_return * (252 ** 0.5)

        return (annual_return - self.risk_free_rate) / annual_std if annual_std > 0 else 0.0

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation version of Sharpe)."""
        if len(returns) < 2:
            return 0.0

        avg_return = statistics.mean(returns)

        # Calculate downside deviation
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return 0.0

        downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0

        if downside_std == 0:
            return 0.0

        # Annualize
        annual_return = avg_return * 252
        annual_downside_std = downside_std * (252 ** 0.5)

        return (annual_return - self.risk_free_rate) / annual_downside_std if annual_downside_std > 0 else 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.pnl_history) < 2:
            return 0.0

        peak = 0
        max_drawdown = 0
        cumulative = 0

        for record in self.pnl_history:
            cumulative += record.daily_pnl
            if cumulative > peak:
                peak = cumulative
            elif peak > 0:
                drawdown = (peak - cumulative) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return max_drawdown

    def _calculate_win_rate(self, pnls: List[float]) -> float:
        """Calculate win rate."""
        if not pnls:
            return 0.0

        winning_trades = len([p for p in pnls if p > 0])
        return winning_trades / len(pnls)

    def _calculate_avg_win(self, pnls: List[float]) -> float:
        """Calculate average winning trade."""
        winning_pnls = [p for p in pnls if p > 0]
        return statistics.mean(winning_pnls) if winning_pnls else 0.0

    def _calculate_avg_loss(self, pnls: List[float]) -> float:
        """Calculate average losing trade."""
        losing_pnls = [p for p in pnls if p < 0]
        return statistics.mean(losing_pnls) if losing_pnls else 0.0

    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        total_profits = sum(p for p in pnls if p > 0)
        total_losses = abs(sum(p for p in pnls if p < 0))

        return total_profits / total_losses if total_losses > 0 else 0.0

    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) < 2:
            return 0.0

        annual_return = statistics.mean(returns) * 252
        max_drawdown = self._calculate_max_drawdown()

        return annual_return / max_drawdown if max_drawdown > 0 else 0.0

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0,
            'calculated_at': datetime.utcnow().isoformat()
        }

    async def get_pnl_history(self, days: int = 30) -> Dict[str, Any]:
        """
        Get P&L history for charting.

        Args:
            days: Number of days of history to retrieve

        Returns:
            P&L history data
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Filter history for the requested period
            filtered_history = [
                record for record in self.pnl_history
                if record.timestamp > cutoff_date
            ]

            if not filtered_history:
                return {
                    'data_points': 0,
                    'start_date': cutoff_date.isoformat(),
                    'end_date': datetime.utcnow().isoformat(),
                    'pnl_data': []
                }

            # Calculate cumulative P&L
            cumulative_pnl = 0
            chart_data = []

            for record in filtered_history:
                cumulative_pnl += record.daily_pnl
                chart_data.append({
                    'date': record.timestamp.date().isoformat(),
                    'daily_pnl': round(record.daily_pnl, 2),
                    'cumulative_pnl': round(cumulative_pnl, 2),
                    'daily_return': round(record.daily_return, 4)
                })

            return {
                'data_points': len(chart_data),
                'start_date': filtered_history[0].timestamp.isoformat(),
                'end_date': filtered_history[-1].timestamp.isoformat(),
                'pnl_data': chart_data
            }

        except Exception as e:
            logger.error(f"Error getting P&L history: {e}")
            return {
                'data_points': 0,
                'start_date': cutoff_date.isoformat(),
                'end_date': datetime.utcnow().isoformat(),
                'pnl_data': []
            }

    async def get_daily_returns_distribution(self) -> Dict[str, Any]:
        """
        Get daily returns distribution for analysis.

        Returns:
            Returns distribution data
        """
        try:
            if len(self.pnl_history) < 2:
                return {'distribution': [], 'statistics': {}}

            daily_returns = [record.daily_return for record in self.pnl_history]

            # Calculate distribution bins
            min_return = min(daily_returns)
            max_return = max(daily_returns)
            bin_count = min(20, len(daily_returns) // 5)

            if bin_count < 2:
                return {'distribution': [], 'statistics': {}}

            bin_size = (max_return - min_return) / bin_count

            distribution = []
            for i in range(bin_count):
                bin_start = min_return + i * bin_size
                bin_end = min_return + (i + 1) * bin_size

                count = len([r for r in daily_returns if bin_start <= r < bin_end])
                distribution.append({
                    'bin_start': round(bin_start, 4),
                    'bin_end': round(bin_end, 4),
                    'count': count,
                    'percentage': round(count / len(daily_returns) * 100, 2)
                })

            # Calculate statistics
            stats = {
                'mean': round(statistics.mean(daily_returns), 6),
                'median': round(statistics.median(daily_returns), 6),
                'std_dev': round(statistics.stdev(daily_returns), 6),
                'min': round(min_return, 6),
                'max': round(max_return, 6),
                'skewness': self._calculate_skewness(daily_returns),
                'kurtosis': self._calculate_kurtosis(daily_returns)
            }

            return {
                'distribution': distribution,
                'statistics': stats
            }

        except Exception as e:
            logger.error(f"Error calculating returns distribution: {e}")
            return {'distribution': [], 'statistics': {}}

    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of returns."""
        if len(data) < 3:
            return 0.0

        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        n = len(data)

        if std_val == 0:
            return 0.0

        skewness = sum(((x - mean_val) / std_val) ** 3 for x in data) / n
        return round(skewness, 6)

    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of returns."""
        if len(data) < 4:
            return 0.0

        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        n = len(data)

        if std_val == 0:
            return 0.0

        kurtosis = sum(((x - mean_val) / std_val) ** 4 for x in data) / n - 3
        return round(kurtosis, 6)

    async def export_pnl_data(self, format: str = 'json') -> str:
        """
        Export P&L data in specified format.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Exported data as string
        """
        try:
            if format.lower() == 'csv':
                # CSV export
                csv_lines = ['Date,Daily P&L,Total P&L,Daily Return,Realized P&L,Unrealized P&L']

                for record in self.pnl_history:
                    csv_lines.append(
                        f"{record.timestamp.date()},"
                        f"{record.daily_pnl},"
                        f"{record.total_pnl},"
                        f"{record.daily_return},"
                        f"{record.realized_pnl},"
                        f"{record.unrealized_pnl}"
                    )

                return '\n'.join(csv_lines)

            else:
                # JSON export
                export_data = {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'data_points': len(self.pnl_history),
                    'pnl_records': [
                        {
                            'timestamp': record.timestamp.isoformat(),
                            'daily_pnl': record.daily_pnl,
                            'total_pnl': record.total_pnl,
                            'daily_return': record.daily_return,
                            'realized_pnl': record.realized_pnl,
                            'unrealized_pnl': record.unrealized_pnl,
                            'commission': record.commission,
                            'slippage': record.slippage
                        }
                        for record in self.pnl_history
                    ]
                }

                return json.dumps(export_data, indent=2)

        except Exception as e:
            logger.error(f"Error exporting P&L data: {e}")
            return ""

    async def clear_history(self) -> bool:
        """
        Clear P&L history.

        Returns:
            True if cleared successfully
        """
        try:
            self.pnl_history.clear()
            self.performance_cache.clear()
            self.last_cache_update = None
            logger.info("P&L history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing P&L history: {e}")
            return False

# Global P&L monitor instance
pnl_monitor = PnLMonitor()
