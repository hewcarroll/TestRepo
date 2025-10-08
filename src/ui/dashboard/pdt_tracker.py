"""
PDT Tracker Module for PDT Trading Bot Admin UI

This module provides Pattern Day Trader compliance tracking, volume monitoring,
and day trade counting for regulatory compliance.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class DayTradeRecord:
    """Day trade record structure."""
    date: date
    symbol: str
    entry_time: datetime
    exit_time: datetime
    quantity: float
    is_day_trade: bool
    rationale: str = ""

@dataclass
class VolumeRecord:
    """Daily volume record structure."""
    date: date
    total_volume: float
    trade_count: int
    symbols_traded: List[str]
    largest_trade: float

@dataclass
class PDTStatus:
    """PDT status structure."""
    is_pdt_account: bool
    day_trades_used: int
    day_trades_remaining: int
    volume_towards_threshold: float
    threshold_remaining: float
    reset_date: date
    status: str  # 'compliant', 'warning', 'violation'
    last_updated: datetime

class PDTTracker:
    """Tracks PDT compliance and volume requirements."""

    def __init__(self):
        """Initialize PDT tracker."""
        self.day_trades: List[DayTradeRecord] = []
        self.volume_history: List[VolumeRecord] = []
        self.pdt_threshold = 25000.00  # $25,000 threshold
        self.max_day_trades = 3  # Maximum day trades per 5-day period
        self.cache_timeout = 300  # 5 minutes
        self.status_cache: Optional[PDTStatus] = None
        self.last_cache_update = None

    async def record_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Record a trade for PDT compliance tracking.

        Args:
            trade_data: Trade data dictionary

        Returns:
            True if recorded successfully
        """
        try:
            # Parse trade data
            symbol = trade_data.get('symbol', '')
            quantity = float(trade_data.get('quantity', 0))
            price = float(trade_data.get('price', 0))
            timestamp = datetime.fromisoformat(trade_data.get('timestamp', datetime.utcnow().isoformat()))
            trade_value = quantity * price

            # Check if this is a day trade (buy and sell same symbol same day)
            is_day_trade = await self._is_day_trade(symbol, timestamp.date())

            # Create day trade record
            day_trade = DayTradeRecord(
                date=timestamp.date(),
                symbol=symbol,
                entry_time=timestamp,
                exit_time=timestamp,  # For now, assume immediate execution
                quantity=quantity,
                is_day_trade=is_day_trade,
                rationale=trade_data.get('rationale', '')
            )

            self.day_trades.append(day_trade)

            # Update volume tracking
            await self._update_volume_record(timestamp.date(), trade_value, symbol)

            # Clear cache when new trade is recorded
            self.status_cache = None
            self.last_cache_update = None

            logger.info(f"Recorded trade for PDT tracking: {symbol} {quantity} @ {price} (Day trade: {is_day_trade})")
            return True

        except Exception as e:
            logger.error(f"Error recording trade for PDT tracking: {e}")
            return False

    async def _is_day_trade(self, symbol: str, trade_date: date) -> bool:
        """
        Check if a trade qualifies as a day trade.

        Args:
            symbol: Trading symbol
            trade_date: Date of the trade

        Returns:
            True if this is a day trade
        """
        try:
            # Get all trades for this symbol on this date
            day_trades_for_symbol = [
                trade for trade in self.day_trades
                if trade.symbol == symbol and trade.date == trade_date
            ]

            if len(day_trades_for_symbol) < 2:
                return False

            # Check if there are both buy and sell trades on the same day
            sides = [trade.quantity > 0 for trade in day_trades_for_symbol]  # Simplified: positive quantity = buy

            # If we have both positive and negative quantities, it's likely a day trade
            has_buys = any(sides)
            has_sells = any(not side for side in sides)

            return has_buys and has_sells

        except Exception as e:
            logger.error(f"Error checking day trade status: {e}")
            return False

    async def _update_volume_record(self, trade_date: date, trade_value: float, symbol: str) -> None:
        """
        Update volume record for a trading day.

        Args:
            trade_date: Date of trading
            trade_value: Value of the trade
            symbol: Trading symbol
        """
        try:
            # Find existing volume record for the day
            volume_record = next(
                (record for record in self.volume_history if record.date == trade_date),
                None
            )

            if volume_record:
                # Update existing record
                volume_record.total_volume += trade_value
                volume_record.trade_count += 1
                if symbol not in volume_record.symbols_traded:
                    volume_record.symbols_traded.append(symbol)
                if trade_value > volume_record.largest_trade:
                    volume_record.largest_trade = trade_value
            else:
                # Create new volume record
                new_record = VolumeRecord(
                    date=trade_date,
                    total_volume=trade_value,
                    trade_count=1,
                    symbols_traded=[symbol],
                    largest_trade=trade_value
                )
                self.volume_history.append(new_record)

            # Keep only last 90 days of volume history
            cutoff_date = date.today() - timedelta(days=90)
            self.volume_history = [
                record for record in self.volume_history
                if record.date >= cutoff_date
            ]

        except Exception as e:
            logger.error(f"Error updating volume record: {e}")

    async def get_pdt_status(self) -> PDTStatus:
        """
        Get current PDT status.

        Returns:
            Current PDT status information
        """
        try:
            # Check cache first
            if (self.status_cache and self.last_cache_update and
                (datetime.utcnow() - self.last_cache_update).seconds < self.cache_timeout):
                return self.status_cache

            # Calculate current status
            today = date.today()

            # Count day trades in current 5-day period
            five_days_ago = today - timedelta(days=4)
            recent_day_trades = [
                trade for trade in self.day_trades
                if trade.date >= five_days_ago and trade.is_day_trade
            ]

            day_trades_used = len(recent_day_trades)
            day_trades_remaining = max(0, self.max_day_trades - day_trades_used)

            # Calculate volume towards threshold
            current_month_volume = sum(
                record.total_volume for record in self.volume_history
                if record.date >= today.replace(day=1)  # Start of current month
            )

            volume_towards_threshold = min(current_month_volume, self.pdt_threshold)
            threshold_remaining = max(0, self.pdt_threshold - volume_towards_threshold)

            # Determine if account is PDT
            is_pdt_account = day_trades_used >= self.max_day_trades or volume_towards_threshold >= self.pdt_threshold

            # Calculate next reset date (next trading day after 5-day period)
            next_reset = today + timedelta(days=1)
            while next_reset.weekday() >= 5:  # Skip weekends
                next_reset += timedelta(days=1)

            # Determine status
            if day_trades_used >= self.max_day_trades:
                status = "violation"
            elif day_trades_used >= 2:
                status = "warning"
            elif volume_towards_threshold >= (self.pdt_threshold * 0.8):
                status = "warning"
            else:
                status = "compliant"

            pdt_status = PDTStatus(
                is_pdt_account=is_pdt_account,
                day_trades_used=day_trades_used,
                day_trades_remaining=day_trades_remaining,
                volume_towards_threshold=round(volume_towards_threshold, 2),
                threshold_remaining=round(threshold_remaining, 2),
                reset_date=next_reset,
                status=status,
                last_updated=datetime.utcnow()
            )

            # Cache the result
            self.status_cache = pdt_status
            self.last_cache_update = datetime.utcnow()

            return pdt_status

        except Exception as e:
            logger.error(f"Error getting PDT status: {e}")
            # Return default status on error
            return PDTStatus(
                is_pdt_account=False,
                day_trades_used=0,
                day_trades_remaining=self.max_day_trades,
                volume_towards_threshold=0.0,
                threshold_remaining=self.pdt_threshold,
                reset_date=date.today() + timedelta(days=1),
                status="unknown",
                last_updated=datetime.utcnow()
            )

    async def get_volume_progress(self, days: int = 30) -> Dict[str, Any]:
        """
        Get volume progress over time.

        Args:
            days: Number of days to analyze

        Returns:
            Volume progress data
        """
        try:
            cutoff_date = date.today() - timedelta(days=days)
            relevant_volume = [
                record for record in self.volume_history
                if record.date >= cutoff_date
            ]

            if not relevant_volume:
                return {
                    'days_analyzed': days,
                    'total_volume': 0.0,
                    'daily_volume': [],
                    'progress_percentage': 0.0,
                    'target_threshold': self.pdt_threshold
                }

            # Calculate cumulative volume
            cumulative_volume = 0
            daily_data = []

            for record in sorted(relevant_volume, key=lambda x: x.date):
                cumulative_volume += record.total_volume
                daily_data.append({
                    'date': record.date.isoformat(),
                    'daily_volume': round(record.total_volume, 2),
                    'cumulative_volume': round(cumulative_volume, 2),
                    'trade_count': record.trade_count,
                    'symbols_traded': record.symbols_traded
                })

            progress_percentage = min(100, (cumulative_volume / self.pdt_threshold) * 100)

            return {
                'days_analyzed': days,
                'total_volume': round(cumulative_volume, 2),
                'daily_volume': daily_data,
                'progress_percentage': round(progress_percentage, 2),
                'target_threshold': self.pdt_threshold,
                'remaining_threshold': round(max(0, self.pdt_threshold - cumulative_volume), 2)
            }

        except Exception as e:
            logger.error(f"Error getting volume progress: {e}")
            return {
                'days_analyzed': days,
                'total_volume': 0.0,
                'daily_volume': [],
                'progress_percentage': 0.0,
                'target_threshold': self.pdt_threshold
            }

    async def get_day_trade_history(self, days: int = 30) -> Dict[str, Any]:
        """
        Get day trade history for compliance review.

        Args:
            days: Number of days to analyze

        Returns:
            Day trade history data
        """
        try:
            cutoff_date = date.today() - timedelta(days=days)
            recent_trades = [
                trade for trade in self.day_trades
                if trade.date >= cutoff_date
            ]

            # Group by date
            daily_trades = {}
            for trade in recent_trades:
                date_str = trade.date.isoformat()
                if date_str not in daily_trades:
                    daily_trades[date_str] = []

                daily_trades[date_str].append({
                    'symbol': trade.symbol,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat(),
                    'quantity': trade.quantity,
                    'is_day_trade': trade.is_day_trade,
                    'rationale': trade.rationale
                })

            return {
                'days_analyzed': days,
                'total_day_trades': len([t for t in recent_trades if t.is_day_trade]),
                'daily_breakdown': daily_trades,
                'start_date': cutoff_date.isoformat(),
                'end_date': date.today().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting day trade history: {e}")
            return {
                'days_analyzed': days,
                'total_day_trades': 0,
                'daily_breakdown': {},
                'start_date': cutoff_date.isoformat(),
                'end_date': date.today().isoformat()
            }

    async def get_compliance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.

        Returns:
            Compliance report data
        """
        try:
            pdt_status = await self.get_pdt_status()
            volume_progress = await self.get_volume_progress(30)
            day_trade_history = await self.get_day_trade_history(30)

            # Calculate compliance score (0-100)
            compliance_score = 100

            # Deduct points for PDT violations
            if pdt_status.status == "violation":
                compliance_score -= 50
            elif pdt_status.status == "warning":
                compliance_score -= 25

            # Deduct points for high day trade usage
            day_trade_usage = pdt_status.day_trades_used / self.max_day_trades
            compliance_score -= int(day_trade_usage * 20)

            # Deduct points for volume threshold proximity
            volume_proximity = pdt_status.volume_towards_threshold / self.pdt_threshold
            if volume_proximity > 0.8:
                compliance_score -= int((volume_proximity - 0.8) * 50)

            compliance_score = max(0, min(100, compliance_score))

            return {
                'pdt_status': {
                    'is_pdt_account': pdt_status.is_pdt_account,
                    'day_trades_used': pdt_status.day_trades_used,
                    'day_trades_remaining': pdt_status.day_trades_remaining,
                    'volume_towards_threshold': pdt_status.volume_towards_threshold,
                    'threshold_remaining': pdt_status.threshold_remaining,
                    'reset_date': pdt_status.reset_date.isoformat(),
                    'status': pdt_status.status
                },
                'volume_progress': volume_progress,
                'day_trade_history': day_trade_history,
                'compliance_score': compliance_score,
                'generated_at': datetime.utcnow().isoformat(),
                'recommendations': self._generate_compliance_recommendations(pdt_status, compliance_score)
            }

        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {
                'error': 'Failed to generate compliance report',
                'generated_at': datetime.utcnow().isoformat()
            }

    def _generate_compliance_recommendations(self, pdt_status: PDTStatus, compliance_score: int) -> List[str]:
        """
        Generate compliance recommendations based on current status.

        Args:
            pdt_status: Current PDT status
            compliance_score: Compliance score (0-100)

        Returns:
            List of recommendation strings
        """
        recommendations = []

        try:
            if compliance_score < 50:
                recommendations.append("URGENT: Account is in violation of PDT rules. Cease day trading immediately.")
                recommendations.append("Consider switching to cash account or depositing funds to meet $25,000 requirement.")
            elif compliance_score < 75:
                recommendations.append("WARNING: Approaching PDT limits. Monitor day trades closely.")
                recommendations.append("Consider reducing trading frequency or increasing account size.")

            if pdt_status.day_trades_used >= 2:
                recommendations.append(f"You have used {pdt_status.day_trades_used} day trades. {pdt_status.day_trades_remaining} remaining this period.")

            if pdt_status.volume_towards_threshold > (self.pdt_threshold * 0.8):
                recommendations.append("Volume is approaching PDT threshold. Monitor monthly trading volume.")

            if pdt_status.status == "compliant":
                recommendations.append("Account is in good compliance standing. Continue following trading rules.")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations at this time."]

    async def export_compliance_data(self, format: str = 'json') -> str:
        """
        Export compliance data for record keeping.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Exported compliance data as string
        """
        try:
            compliance_report = await self.get_compliance_report()

            if format.lower() == 'csv':
                # CSV export for compliance records
                csv_lines = [
                    'Date,Metric,Value',
                    f"{datetime.utcnow().date()},PDT Account,{compliance_report['pdt_status']['is_pdt_account']}",
                    f"{datetime.utcnow().date()},Day Trades Used,{compliance_report['pdt_status']['day_trades_used']}",
                    f"{datetime.utcnow().date()},Day Trades Remaining,{compliance_report['pdt_status']['day_trades_remaining']}",
                    f"{datetime.utcnow().date()},Volume Towards Threshold,${compliance_report['pdt_status']['volume_towards_threshold']}",
                    f"{datetime.utcnow().date()},Compliance Score,{compliance_report['compliance_score']}"
                ]

                return '\n'.join(csv_lines)

            else:
                # JSON export
                return json.dumps(compliance_report, indent=2)

        except Exception as e:
            logger.error(f"Error exporting compliance data: {e}")
            return ""

    async def clear_history(self) -> bool:
        """
        Clear all tracking history.

        Returns:
            True if cleared successfully
        """
        try:
            self.day_trades.clear()
            self.volume_history.clear()
            self.status_cache = None
            self.last_cache_update = None
            logger.info("PDT tracking history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing PDT history: {e}")
            return False

    async def get_pdt_alerts(self) -> List[Dict[str, Any]]:
        """
        Get PDT-related alerts and warnings.

        Returns:
            List of alert dictionaries
        """
        try:
            alerts = []
            pdt_status = await self.get_pdt_status()

            # Day trade alerts
            if pdt_status.day_trades_used >= 2:
                severity = "warning" if pdt_status.day_trades_used == 2 else "critical"
                alerts.append({
                    'type': 'day_trade_warning',
                    'severity': severity,
                    'message': f"You have used {pdt_status.day_trades_used} day trades. {pdt_status.day_trades_remaining} remaining.",
                    'timestamp': datetime.utcnow().isoformat()
                })

            # Volume threshold alerts
            volume_percentage = (pdt_status.volume_towards_threshold / self.pdt_threshold) * 100
            if volume_percentage > 80:
                severity = "critical" if volume_percentage > 90 else "warning"
                alerts.append({
                    'type': 'volume_threshold_warning',
                    'severity': severity,
                    'message': f"Monthly volume is {volume_percentage:.1f}% of PDT threshold (${pdt_status.volume_towards_threshold:,.2f} of ${self.pdt_threshold:,.2f}).",
                    'timestamp': datetime.utcnow().isoformat()
                })

            # PDT violation alert
            if pdt_status.status == "violation":
                alerts.append({
                    'type': 'pdt_violation',
                    'severity': 'critical',
                    'message': "Account is in PDT violation. Day trading is restricted.",
                    'timestamp': datetime.utcnow().isoformat()
                })

            return alerts

        except Exception as e:
            logger.error(f"Error getting PDT alerts: {e}")
            return []

# Global PDT tracker instance
pdt_tracker = PDTTracker()
