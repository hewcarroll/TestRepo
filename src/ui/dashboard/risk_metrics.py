"""
Risk Metrics Module for PDT Trading Bot Admin UI

This module provides comprehensive risk analysis, metrics calculation,
and risk monitoring for the trading dashboard.
"""

import asyncio
import json
import math
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetricsData:
    """Risk metrics data structure."""
    timestamp: datetime
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    beta: float
    volatility: float
    exposure: float
    leverage: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

@dataclass
class PositionRisk:
    """Position-specific risk data."""
    symbol: str
    position_value: float
    unrealized_pnl: float
    beta: float
    delta: float
    gamma: float
    theta: float
    vega: float

class RiskMetricsCalculator:
    """Calculates comprehensive risk metrics for the trading portfolio."""

    def __init__(self):
        """Initialize risk calculator."""
        self.risk_history: List[RiskMetricsData] = []
        self.position_risks: Dict[str, PositionRisk] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.cache_timeout = 300  # 5 minutes
        self.metrics_cache: Optional[Dict[str, Any]] = None
        self.last_cache_update = None

        # Risk-free rate for Sharpe/Sortino calculations
        self.risk_free_rate = 0.02  # 2% annual

    async def update_risk_data(self, portfolio_data: Dict[str, Any]) -> None:
        """
        Update risk metrics with latest portfolio data.

        Args:
            portfolio_data: Current portfolio and market data
        """
        try:
            # Extract returns data for calculations
            returns = portfolio_data.get('daily_returns', [])
            positions = portfolio_data.get('positions', [])
            market_data = portfolio_data.get('market_data', {})

            if not returns:
                logger.warning("No returns data available for risk calculation")
                return

            # Calculate current risk metrics
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            volatility = self._calculate_volatility(returns)
            beta = self._calculate_beta(returns, market_data)
            max_drawdown = self._calculate_max_drawdown(returns)

            # Calculate portfolio metrics
            total_exposure = sum(abs(pos.get('market_value', 0)) for pos in positions)
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            leverage = total_exposure / total_value if total_value > 0 else 1.0

            # Calculate ratios
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)

            # Create risk metrics record
            risk_record = RiskMetricsData(
                timestamp=datetime.utcnow(),
                var_95=var_95,
                var_99=var_99,
                beta=beta,
                volatility=volatility,
                exposure=total_exposure,
                leverage=leverage,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio
            )

            self.risk_history.append(risk_record)

            # Keep only last 90 days of risk data
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            self.risk_history = [
                record for record in self.risk_history
                if record.timestamp > cutoff_date
            ]

            # Update position-specific risks
            await self._update_position_risks(positions, market_data)

            # Clear cache
            self.metrics_cache = None
            self.last_cache_update = None

            logger.info(f"Updated risk metrics: VaR95=${var_95}, Volatility={volatility:.4f}, Sharpe={sharpe_ratio:.4f}")

        except Exception as e:
            logger.error(f"Error updating risk data: {e}")

    def _calculate_var(self, returns: List[float], confidence_level: float) -> float:
        """
        Calculate Value at Risk (VaR) at specified confidence level.

        Args:
            returns: List of historical returns
            confidence_level: Confidence level (0.95 for 95% VaR)

        Returns:
            VaR value
        """
        if len(returns) < 30:
            return 0.0

        try:
            # Sort returns in ascending order
            sorted_returns = sorted(returns)

            # Find the index for the confidence level
            index = int(len(sorted_returns) * (1 - confidence_level))

            if index >= len(sorted_returns):
                index = len(sorted_returns) - 1

            # VaR is the negative of the return at the confidence level
            var_return = sorted_returns[index]
            return abs(var_return)

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def _calculate_volatility(self, returns: List[float]) -> float:
        """
        Calculate portfolio volatility (standard deviation of returns).

        Args:
            returns: List of historical returns

        Returns:
            Volatility as decimal
        """
        if len(returns) < 2:
            return 0.0

        try:
            return statistics.stdev(returns)
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def _calculate_beta(self, returns: List[float], market_data: Dict[str, Any]) -> float:
        """
        Calculate portfolio beta against market.

        Args:
            returns: Portfolio returns
            market_data: Market benchmark data

        Returns:
            Portfolio beta
        """
        if len(returns) < 30:
            return 1.0

        try:
            # For simplicity, assume market return is S&P 500
            # In production, this would use actual market data
            market_returns = market_data.get('market_returns', [0.0001] * len(returns))

            if len(market_returns) != len(returns):
                return 1.0

            # Calculate covariance and variance
            covariance = statistics.covariance(returns, market_returns)
            market_variance = statistics.variance(market_returns)

            if covariance is None or market_variance == 0:
                return 1.0

            return covariance / market_variance

        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns.

        Args:
            returns: List of cumulative returns

        Returns:
            Maximum drawdown as decimal
        """
        if len(returns) < 2:
            return 0.0

        try:
            # Convert to cumulative returns if not already
            cumulative = 0
            peak = 0
            max_drawdown = 0

            for ret in returns:
                cumulative += ret
                if cumulative > peak:
                    peak = cumulative
                elif peak > 0:
                    drawdown = (peak - cumulative) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

            return max_drawdown

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: List of returns

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        try:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)

            if std_return == 0:
                return 0.0

            # Annualize (assuming daily returns)
            annual_return = avg_return * 252
            annual_std = std_return * math.sqrt(252)

            return (annual_return - self.risk_free_rate) / annual_std if annual_std > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sortino ratio (downside deviation version of Sharpe).

        Args:
            returns: List of returns

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        try:
            avg_return = statistics.mean(returns)

            # Calculate downside deviation
            downside_returns = [r for r in returns if r < 0]
            if not downside_returns:
                return 0.0

            downside_std = statistics.stdev(downside_returns)

            if downside_std == 0:
                return 0.0

            # Annualize
            annual_return = avg_return * 252
            annual_downside_std = downside_std * math.sqrt(252)

            return (annual_return - self.risk_free_rate) / annual_downside_std if annual_downside_std > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).

        Args:
            returns: List of returns
            max_drawdown: Maximum drawdown

        Returns:
            Calmar ratio
        """
        if len(returns) < 2 or max_drawdown == 0:
            return 0.0

        try:
            annual_return = statistics.mean(returns) * 252
            return annual_return / max_drawdown if max_drawdown > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0

    async def _update_position_risks(self, positions: List[Dict[str, Any]],
                                   market_data: Dict[str, Any]) -> None:
        """
        Update position-specific risk metrics.

        Args:
            positions: Current positions
            market_data: Market data for calculations
        """
        try:
            for position in positions:
                symbol = position.get('symbol', '')
                position_value = position.get('market_value', 0)
                unrealized_pnl = position.get('unrealized_pnl', 0)

                # Get Greeks (simplified calculation)
                # In production, this would use proper options pricing models
                current_price = market_data.get(symbol, {}).get('price', 0)
                delta = 1.0 if position_value > 0 else -1.0  # Simplified delta
                gamma = 0.01  # Simplified gamma
                theta = -0.001  # Simplified theta
                vega = 0.1  # Simplified vega

                position_risk = PositionRisk(
                    symbol=symbol,
                    position_value=position_value,
                    unrealized_pnl=unrealized_pnl,
                    beta=1.0,  # Would be calculated from historical data
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega
                )

                self.position_risks[symbol] = position_risk

        except Exception as e:
            logger.error(f"Error updating position risks: {e}")

    async def get_current_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current comprehensive risk metrics.

        Returns:
            Current risk metrics
        """
        try:
            # Check cache first
            if (self.metrics_cache and self.last_cache_update and
                (datetime.utcnow() - self.last_cache_update).seconds < self.cache_timeout):
                return self.metrics_cache

            if not self.risk_history:
                return self._get_empty_risk_metrics()

            latest = self.risk_history[-1]

            # Get position-specific risks
            position_risks = []
            for symbol, risk in self.position_risks.items():
                position_risks.append({
                    'symbol': risk.symbol,
                    'position_value': round(risk.position_value, 2),
                    'unrealized_pnl': round(risk.unrealized_pnl, 2),
                    'beta': round(risk.beta, 4),
                    'delta': round(risk.delta, 4),
                    'gamma': round(risk.gamma, 4),
                    'theta': round(risk.theta, 4),
                    'vega': round(risk.vega, 4)
                })

            # Calculate stress test scenarios
            stress_tests = await self._calculate_stress_tests()

            # Calculate risk contributions
            risk_contributions = await self._calculate_risk_contributions()

            metrics = {
                'timestamp': latest.timestamp.isoformat(),
                'var_95': round(latest.var_95, 2),
                'var_99': round(latest.var_99, 2),
                'beta': round(latest.beta, 4),
                'volatility': round(latest.volatility, 4),
                'exposure': round(latest.exposure, 2),
                'leverage': round(latest.leverage, 4),
                'max_drawdown': round(latest.max_drawdown, 4),
                'sharpe_ratio': round(latest.sharpe_ratio, 4),
                'sortino_ratio': round(latest.sortino_ratio, 4),
                'calmar_ratio': round(latest.calmar_ratio, 4),
                'position_risks': position_risks,
                'stress_tests': stress_tests,
                'risk_contributions': risk_contributions,
                'risk_level': self._assess_risk_level(latest)
            }

            # Cache results
            self.metrics_cache = metrics
            self.last_cache_update = datetime.utcnow()

            return metrics

        except Exception as e:
            logger.error(f"Error getting current risk metrics: {e}")
            return self._get_empty_risk_metrics()

    def _get_empty_risk_metrics(self) -> Dict[str, Any]:
        """Return empty risk metrics structure."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'var_95': 0.0,
            'var_99': 0.0,
            'beta': 1.0,
            'volatility': 0.0,
            'exposure': 0.0,
            'leverage': 1.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'position_risks': [],
            'stress_tests': {},
            'risk_contributions': {},
            'risk_level': 'unknown'
        }

    async def _calculate_stress_tests(self) -> Dict[str, float]:
        """
        Calculate stress test scenarios.

        Returns:
            Stress test results
        """
        try:
            if not self.risk_history:
                return {}

            latest = self.risk_history[-1]

            # Simple stress test scenarios
            stress_tests = {
                'market_crash_20': latest.exposure * -0.20,  # 20% market crash
                'market_crash_30': latest.exposure * -0.30,  # 30% market crash
                'volatility_spike': latest.exposure * -0.15,  # High volatility scenario
                'liquidity_crisis': latest.exposure * -0.10,  # Liquidity crisis scenario
                'sector_rotation': latest.exposure * -0.08   # Sector rotation scenario
            }

            return {k: round(v, 2) for k, v in stress_tests.items()}

        except Exception as e:
            logger.error(f"Error calculating stress tests: {e}")
            return {}

    async def _calculate_risk_contributions(self) -> Dict[str, float]:
        """
        Calculate risk contribution by position.

        Returns:
            Risk contribution by symbol
        """
        try:
            contributions = {}

            for symbol, risk in self.position_risks.items():
                # Simplified risk contribution based on position size and beta
                position_weight = abs(risk.position_value) / max(sum(
                    abs(r.position_value) for r in self.position_risks.values()
                ), 1)

                risk_contribution = position_weight * risk.beta
                contributions[symbol] = round(risk_contribution, 4)

            return contributions

        except Exception as e:
            logger.error(f"Error calculating risk contributions: {e}")
            return {}

    def _assess_risk_level(self, metrics: RiskMetricsData) -> str:
        """
        Assess overall risk level based on metrics.

        Args:
            metrics: Current risk metrics

        Returns:
            Risk level ('low', 'medium', 'high', 'critical')
        """
        try:
            risk_score = 0

            # VaR assessment
            if metrics.var_95 > 0.15:  # 15% daily VaR
                risk_score += 3
            elif metrics.var_95 > 0.10:
                risk_score += 2
            elif metrics.var_95 > 0.05:
                risk_score += 1

            # Volatility assessment
            if metrics.volatility > 0.30:  # 30% annual volatility
                risk_score += 3
            elif metrics.volatility > 0.20:
                risk_score += 2
            elif metrics.volatility > 0.15:
                risk_score += 1

            # Leverage assessment
            if metrics.leverage > 3.0:
                risk_score += 3
            elif metrics.leverage > 2.0:
                risk_score += 2
            elif metrics.leverage > 1.5:
                risk_score += 1

            # Sharpe ratio assessment (negative score for poor risk-adjusted returns)
            if metrics.sharpe_ratio < 0:
                risk_score += 2
            elif metrics.sharpe_ratio < 0.5:
                risk_score += 1

            # Determine risk level
            if risk_score >= 7:
                return 'critical'
            elif risk_score >= 5:
                return 'high'
            elif risk_score >= 3:
                return 'medium'
            else:
                return 'low'

        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return 'unknown'

    async def get_risk_history(self, days: int = 30) -> Dict[str, Any]:
        """
        Get risk metrics history for charting.

        Args:
            days: Number of days of history

        Returns:
            Risk history data
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            relevant_history = [
                record for record in self.risk_history
                if record.timestamp > cutoff_date
            ]

            if not relevant_history:
                return {
                    'data_points': 0,
                    'start_date': cutoff_date.isoformat(),
                    'end_date': datetime.utcnow().isoformat(),
                    'risk_data': []
                }

            # Prepare chart data
            chart_data = []
            for record in sorted(relevant_history, key=lambda x: x.timestamp):
                chart_data.append({
                    'date': record.timestamp.date().isoformat(),
                    'var_95': round(record.var_95, 4),
                    'var_99': round(record.var_99, 4),
                    'volatility': round(record.volatility, 4),
                    'sharpe_ratio': round(record.sharpe_ratio, 4),
                    'exposure': round(record.exposure, 2)
                })

            return {
                'data_points': len(chart_data),
                'start_date': relevant_history[0].timestamp.isoformat(),
                'end_date': relevant_history[-1].timestamp.isoformat(),
                'risk_data': chart_data
            }

        except Exception as e:
            logger.error(f"Error getting risk history: {e}")
            return {
                'data_points': 0,
                'start_date': cutoff_date.isoformat(),
                'end_date': datetime.utcnow().isoformat(),
                'risk_data': []
            }

    async def export_risk_report(self, format: str = 'json') -> str:
        """
        Export comprehensive risk report.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Risk report as string
        """
        try:
            current_metrics = await self.get_current_risk_metrics()
            risk_history = await self.get_risk_history(30)

            report_data = {
                'generated_at': datetime.utcnow().isoformat(),
                'current_metrics': current_metrics,
                'risk_history': risk_history,
                'summary': {
                    'overall_risk_level': current_metrics.get('risk_level', 'unknown'),
                    'key_concerns': self._identify_key_risk_concerns(current_metrics),
                    'recommendations': self._generate_risk_recommendations(current_metrics)
                }
            }

            if format.lower() == 'csv':
                # CSV export for risk data
                csv_lines = [
                    'Date,VaR95,VaR99,Volatility,Sharpe,Exposure'
                ]

                for record in report_data['risk_history']['risk_data']:
                    csv_lines.append(
                        f"{record['date']},"
                        f"{record['var_95']},"
                        f"{record['var_99']},"
                        f"{record['volatility']},"
                        f"{record['sharpe_ratio']},"
                        f"{record['exposure']}"
                    )

                return '\n'.join(csv_lines)

            else:
                # JSON export
                return json.dumps(report_data, indent=2)

        except Exception as e:
            logger.error(f"Error exporting risk report: {e}")
            return ""

    def _identify_key_risk_concerns(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Identify key risk concerns from metrics.

        Args:
            metrics: Current risk metrics

        Returns:
            List of risk concerns
        """
        concerns = []

        try:
            if metrics.get('var_95', 0) > 0.10:
                concerns.append("High Value at Risk (95%) indicates significant potential losses")

            if metrics.get('volatility', 0) > 0.25:
                concerns.append("High portfolio volatility suggests increased risk")

            if metrics.get('leverage', 1) > 2.0:
                concerns.append("High leverage amplifies both gains and losses")

            if metrics.get('sharpe_ratio', 0) < 0.5:
                concerns.append("Low Sharpe ratio indicates poor risk-adjusted returns")

            if metrics.get('max_drawdown', 0) > 0.20:
                concerns.append("High maximum drawdown suggests significant historical losses")

            if not concerns:
                concerns.append("No major risk concerns identified")

            return concerns

        except Exception as e:
            logger.error(f"Error identifying risk concerns: {e}")
            return ["Unable to analyze risk concerns"]

    def _generate_risk_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate risk management recommendations.

        Args:
            metrics: Current risk metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        try:
            risk_level = metrics.get('risk_level', 'unknown')

            if risk_level in ['high', 'critical']:
                recommendations.append("Consider reducing position sizes to lower overall exposure")
                recommendations.append("Review and potentially reduce leverage usage")

            if metrics.get('volatility', 0) > 0.25:
                recommendations.append("High volatility detected - consider implementing stricter stop-loss orders")

            if metrics.get('sharpe_ratio', 0) < 0:
                recommendations.append("Negative Sharpe ratio - review strategy effectiveness and risk management")

            if metrics.get('leverage', 1) > 2.0:
                recommendations.append("High leverage detected - consider deleveraging to reduce risk")

            recommendations.append("Regularly monitor risk metrics and adjust positions as needed")
            recommendations.append("Maintain diversified portfolio to spread risk across assets")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations at this time"]

    async def clear_risk_history(self) -> bool:
        """
        Clear risk history data.

        Returns:
            True if cleared successfully
        """
        try:
            self.risk_history.clear()
            self.position_risks.clear()
            self.metrics_cache = None
            self.last_cache_update = None
            logger.info("Risk history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing risk history: {e}")
            return False

# Global risk metrics calculator instance
risk_calculator = RiskMetricsCalculator()
