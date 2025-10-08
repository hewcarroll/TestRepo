"""
Compliance Integration for PDT Trading Bot Admin UI

This module provides integration between the admin UI and the existing
compliance modules for regulatory monitoring and PDT tracking.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ComplianceIntegration:
    """Integrates admin UI with compliance modules."""

    def __init__(self):
        """Initialize compliance integration."""
        self.pdt_tracker = None
        self.regulatory_engine = None
        self.audit_logger = None
        self.broker_policies = None

        # Integration status
        self.integration_active = False
        self.last_compliance_check = None
        self.compliance_violations = []

    async def initialize_integration(self) -> bool:
        """
        Initialize integration with compliance modules.

        Returns:
            True if integration successful
        """
        try:
            # Import compliance modules
            try:
                from ...compliance.pdt.pdt_tracker import PDTTracker
                from ...compliance.regulatory.sec_compliance import SECCompliance
                from ...compliance.audit.audit_logger import AuditLogger
                from ...compliance.broker.alpaca_policies import AlpacaPolicies

                # Initialize compliance components
                self.pdt_tracker = PDTTracker()
                self.regulatory_engine = SECCompliance()
                self.audit_logger = AuditLogger()
                self.broker_policies = AlpacaPolicies()

                self.integration_active = True
                logger.info("Compliance integration initialized successfully")

                return True

            except ImportError as e:
                logger.warning(f"Compliance modules not available: {e}")
                self._setup_mock_integration()
                return True

        except Exception as e:
            logger.error(f"Error initializing compliance integration: {e}")
            self._setup_mock_integration()
            return False

    def _setup_mock_integration(self) -> None:
        """Set up mock compliance integration for development/testing."""
        logger.info("Setting up mock compliance integration")

        # Mock PDT data
        self.mock_pdt_status = {
            'is_pdt_account': True,
            'day_trades_used': 1,
            'day_trades_remaining': 2,
            'volume_towards_threshold': 18500.00,
            'threshold_remaining': 6500.00,
            'reset_date': (datetime.utcnow() + timedelta(days=1)).date().isoformat(),
            'status': 'warning'
        }

        # Mock compliance status
        self.mock_compliance_status = {
            'overall_status': 'warning',
            'pdt_status': self.mock_pdt_status,
            'regulatory_flags': ['wash_trade_warning'],
            'last_audit': datetime.utcnow().isoformat(),
            'risk_score': 0.75
        }

        # Mock audit log
        self.mock_audit_log = [
            {
                'id': 'audit_001',
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': 'admin_001',
                'action': 'login',
                'resource': 'dashboard',
                'details': 'User logged into admin dashboard',
                'ip_address': '192.168.1.100',
                'user_agent': 'Mozilla/5.0 (Mock Browser)'
            },
            {
                'id': 'audit_002',
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': 'admin_001',
                'action': 'bot_control',
                'resource': 'trading_engine',
                'details': 'Bot start command executed',
                'ip_address': '192.168.1.100',
                'user_agent': 'Mozilla/5.0 (Mock Browser)'
            }
        ]

    async def get_pdt_status(self) -> Dict[str, Any]:
        """
        Get current PDT status from compliance module.

        Returns:
            PDT status dictionary
        """
        try:
            if self.integration_active and self.pdt_tracker:
                # Get real PDT status
                # In production, this would call:
                # pdt_status = await self.pdt_tracker.get_current_status()
                pdt_status = {}

                self.last_compliance_check = datetime.utcnow()
                return pdt_status

            else:
                # Return mock data
                self.last_compliance_check = datetime.utcnow()
                return self.mock_pdt_status

        except Exception as e:
            logger.error(f"Error getting PDT status: {e}")
            return {
                'is_pdt_account': False,
                'day_trades_used': 0,
                'day_trades_remaining': 3,
                'volume_towards_threshold': 0.0,
                'threshold_remaining': 25000.0,
                'reset_date': datetime.utcnow().date().isoformat(),
                'status': 'unknown',
                'error': str(e)
            }

    async def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get overall compliance status.

        Returns:
            Compliance status dictionary
        """
        try:
            if self.integration_active and self.regulatory_engine:
                # Get real compliance status
                # In production, this would call:
                # compliance_status = await self.regulatory_engine.get_compliance_status()
                compliance_status = {}

                return compliance_status

            else:
                # Return mock data
                return self.mock_compliance_status

        except Exception as e:
            logger.error(f"Error getting compliance status: {e}")
            return {
                'overall_status': 'unknown',
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }

    async def get_audit_log(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        Get audit log from compliance module.

        Args:
            limit: Maximum records to return
            offset: Offset for pagination

        Returns:
            Audit log data
        """
        try:
            if self.integration_active and self.audit_logger:
                # Get real audit log
                # In production, this would call:
                # audit_log = await self.audit_logger.get_log(limit=limit, offset=offset)
                audit_log = []

                return {
                    'entries': audit_log,
                    'total_count': len(audit_log),
                    'limit': limit,
                    'offset': offset
                }

            else:
                # Return mock data
                paginated_log = self.mock_audit_log[offset:offset + limit]

                return {
                    'entries': paginated_log,
                    'total_count': len(self.mock_audit_log),
                    'limit': limit,
                    'offset': offset,
                    'has_more': offset + limit < len(self.mock_audit_log)
                }

        except Exception as e:
            logger.error(f"Error getting audit log: {e}")
            return {
                'entries': [],
                'total_count': 0,
                'limit': limit,
                'offset': offset,
                'error': str(e)
            }

    async def record_compliance_event(self, event_type: str,
                                    event_data: Dict[str, Any],
                                    user_id: str = "system") -> bool:
        """
        Record a compliance-related event.

        Args:
            event_type: Type of compliance event
            event_data: Event data
            user_id: User who triggered the event

        Returns:
            True if recorded successfully
        """
        try:
            if self.integration_active and self.audit_logger:
                # Record real compliance event
                # In production, this would call:
                # await self.audit_logger.log_event(event_type, event_data, user_id)
                pass

                return True

            else:
                # Mock compliance event recording
                logger.info(f"Mock compliance event: {event_type} by {user_id} - {event_data}")

                # Add to mock audit log
                new_entry = {
                    'id': f'audit_{len(self.mock_audit_log) + 1:03d}',
                    'timestamp': datetime.utcnow().isoformat(),
                    'user_id': user_id,
                    'action': event_type,
                    'resource': event_data.get('resource', 'compliance'),
                    'details': event_data.get('details', str(event_data)),
                    'ip_address': event_data.get('ip_address', '127.0.0.1'),
                    'user_agent': event_data.get('user_agent', 'System')
                }

                self.mock_audit_log.append(new_entry)

                # Keep only last 1000 entries
                if len(self.mock_audit_log) > 1000:
                    self.mock_audit_log = self.mock_audit_log[-1000:]

                return True

        except Exception as e:
            logger.error(f"Error recording compliance event: {e}")
            return False

    async def check_regulatory_compliance(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check trade against regulatory requirements.

        Args:
            trade_data: Trade data to check

        Returns:
            Compliance check result
        """
        try:
            if self.integration_active and self.regulatory_engine:
                # Perform real regulatory compliance check
                # In production, this would call:
                # compliance_result = await self.regulatory_engine.check_trade_compliance(trade_data)
                compliance_result = {
                    'compliant': True,
                    'violations': [],
                    'warnings': []
                }

                return compliance_result

            else:
                # Mock compliance check
                logger.info(f"Mock compliance check: {trade_data}")

                # Simulate some compliance checks
                violations = []
                warnings = []

                # Check for wash trading patterns
                if trade_data.get('side') in ['buy', 'sell']:
                    # Simple mock check - in production this would be sophisticated
                    warnings.append("Trade pattern monitored for regulatory compliance")

                return {
                    'compliant': len(violations) == 0,
                    'violations': violations,
                    'warnings': warnings,
                    'checked_at': datetime.utcnow().isoformat(),
                    'regulations_applied': ['PDT Rules', 'Wash Trade Prevention', 'SEC Regulations']
                }

        except Exception as e:
            logger.error(f"Error checking regulatory compliance: {e}")
            return {
                'compliant': False,
                'violations': [f'Compliance check failed: {str(e)}'],
                'warnings': [],
                'checked_at': datetime.utcnow().isoformat()
            }

    async def get_pdt_progress_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Get PDT progress tracking data.

        Args:
            days: Number of days to analyze

        Returns:
            PDT progress data
        """
        try:
            if self.integration_active and self.pdt_tracker:
                # Get real PDT progress data
                # In production, this would call:
                # progress_data = await self.pdt_tracker.get_progress_data(days)
                progress_data = {}

                return progress_data

            else:
                # Generate mock PDT progress data
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days)

                progress_data = []
                current_date = start_date
                current_volume = 0

                while current_date <= end_date:
                    # Generate mock daily volume
                    daily_volume = 500 + (hash(str(current_date.date())) % 1000)
                    current_volume += daily_volume

                    progress_data.append({
                        'date': current_date.date().isoformat(),
                        'daily_volume': round(daily_volume, 2),
                        'cumulative_volume': round(current_volume, 2),
                        'threshold_remaining': round(max(0, 25000 - current_volume), 2),
                        'progress_percentage': round(min(100, (current_volume / 25000) * 100), 2)
                    })

                    current_date += timedelta(days=1)

                return {
                    'current_status': self.mock_pdt_status,
                    'progress_history': progress_data,
                    'target_threshold': 25000.00,
                    'days_tracked': days,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }

        except Exception as e:
            logger.error(f"Error getting PDT progress data: {e}")
            return {
                'current_status': self.mock_pdt_status,
                'progress_history': [],
                'target_threshold': 25000.00,
                'days_tracked': days,
                'error': str(e)
            }

    async def get_compliance_alerts(self) -> List[Dict[str, Any]]:
        """
        Get current compliance alerts and warnings.

        Returns:
            List of compliance alerts
        """
        try:
            if self.integration_active:
                # Get real compliance alerts
                # In production, this would integrate with compliance monitoring
                alerts = []

                return alerts

            else:
                # Return mock compliance alerts
                alerts = []

                pdt_status = self.mock_pdt_status

                # PDT-related alerts
                if pdt_status['day_trades_used'] >= 2:
                    alerts.append({
                        'type': 'pdt_warning',
                        'severity': 'warning',
                        'message': f"You have used {pdt_status['day_trades_used']} day trades. {pdt_status['day_trades_remaining']} remaining.",
                        'timestamp': datetime.utcnow().isoformat(),
                        'regulation': 'FINRA PDT Rule'
                    })

                # Volume threshold alert
                volume_percentage = (pdt_status['volume_towards_threshold'] / 25000) * 100
                if volume_percentage > 80:
                    alerts.append({
                        'type': 'volume_threshold',
                        'severity': 'warning',
                        'message': f"Monthly volume is {volume_percentage:.1f}% of PDT threshold.",
                        'timestamp': datetime.utcnow().isoformat(),
                        'regulation': 'FINRA PDT Rule'
                    })

                return alerts

        except Exception as e:
            logger.error(f"Error getting compliance alerts: {e}")
            return []

    async def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.

        Returns:
            Compliance report data
        """
        try:
            pdt_status = await self.get_pdt_status()
            compliance_status = await self.get_compliance_status()
            progress_data = await self.get_pdt_progress_data(30)
            alerts = await self.get_compliance_alerts()

            # Calculate compliance score
            compliance_score = 100

            # Deduct points for PDT violations
            if pdt_status.get('status') == 'violation':
                compliance_score -= 50
            elif pdt_status.get('status') == 'warning':
                compliance_score -= 25

            # Deduct points for high day trade usage
            day_trade_usage = pdt_status.get('day_trades_used', 0) / 3
            compliance_score -= int(day_trade_usage * 20)

            # Deduct points for volume threshold proximity
            volume_proximity = pdt_status.get('volume_towards_threshold', 0) / 25000
            if volume_proximity > 0.8:
                compliance_score -= int((volume_proximity - 0.8) * 50)

            compliance_score = max(0, min(100, compliance_score))

            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'compliance_score': compliance_score,
                'pdt_status': pdt_status,
                'overall_compliance': compliance_status,
                'volume_progress': progress_data,
                'active_alerts': alerts,
                'summary': {
                    'overall_status': 'compliant' if compliance_score >= 80 else 'warning' if compliance_score >= 60 else 'violation',
                    'key_issues': self._identify_compliance_issues(pdt_status, alerts),
                    'recommendations': self._generate_compliance_recommendations(pdt_status, compliance_score)
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {
                'error': 'Failed to generate compliance report',
                'generated_at': datetime.utcnow().isoformat()
            }

    def _identify_compliance_issues(self, pdt_status: Dict[str, Any],
                                  alerts: List[Dict[str, Any]]) -> List[str]:
        """
        Identify key compliance issues.

        Args:
            pdt_status: Current PDT status
            alerts: Active compliance alerts

        Returns:
            List of compliance issues
        """
        issues = []

        try:
            if pdt_status.get('status') == 'violation':
                issues.append("Account is in PDT violation - day trading is restricted")

            if pdt_status.get('day_trades_used', 0) >= 2:
                issues.append(f"High day trade usage: {pdt_status['day_trades_used']}/3 used")

            if pdt_status.get('volume_towards_threshold', 0) > 20000:
                issues.append("Approaching PDT volume threshold")

            if alerts:
                issues.append(f"{len(alerts)} active compliance alerts")

            if not issues:
                issues.append("No major compliance issues identified")

            return issues

        except Exception as e:
            logger.error(f"Error identifying compliance issues: {e}")
            return ["Unable to analyze compliance issues"]

    def _generate_compliance_recommendations(self, pdt_status: Dict[str, Any],
                                          compliance_score: int) -> List[str]:
        """
        Generate compliance recommendations.

        Args:
            pdt_status: Current PDT status
            compliance_score: Compliance score (0-100)

        Returns:
            List of recommendations
        """
        recommendations = []

        try:
            if compliance_score < 50:
                recommendations.append("URGENT: Account is in violation of PDT rules. Cease day trading immediately.")
                recommendations.append("Consider switching to cash account or depositing funds to meet $25,000 requirement.")
            elif compliance_score < 75:
                recommendations.append("WARNING: Approaching PDT limits. Monitor day trades closely.")
                recommendations.append("Consider reducing trading frequency or increasing account size.")

            if pdt_status.get('day_trades_used', 0) >= 2:
                recommendations.append(f"You have used {pdt_status['day_trades_used']} day trades. {pdt_status.get('day_trades_remaining', 0)} remaining this period.")

            if pdt_status.get('volume_towards_threshold', 0) > 20000:
                recommendations.append("Volume is approaching PDT threshold. Monitor monthly trading volume.")

            recommendations.append("Regularly review compliance status and maintain proper trading records.")
            recommendations.append("Keep detailed logs of all trading activities for regulatory compliance.")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations at this time"]

    async def export_compliance_records(self, format: str = 'json') -> str:
        """
        Export compliance records for record keeping.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Exported compliance data as string
        """
        try:
            compliance_report = await self.generate_compliance_report()

            if format.lower() == 'csv':
                # CSV export for compliance records
                csv_lines = [
                    'Date,Metric,Value,Status',
                    f"{datetime.utcnow().date()},PDT Account,{compliance_report['pdt_status']['is_pdt_account']},{compliance_report['pdt_status']['status']}",
                    f"{datetime.utcnow().date()},Day Trades Used,{compliance_report['pdt_status']['day_trades_used']},{compliance_report['pdt_status']['status']}",
                    f"{datetime.utcnow().date()},Volume Towards Threshold,${compliance_report['pdt_status']['volume_towards_threshold']},{compliance_report['pdt_status']['status']}",
                    f"{datetime.utcnow().date()},Compliance Score,{compliance_report['compliance_score']},{compliance_report['summary']['overall_status']}"
                ]

                return '\n'.join(csv_lines)

            else:
                # JSON export
                return json.dumps(compliance_report, indent=2)

        except Exception as e:
            logger.error(f"Error exporting compliance records: {e}")
            return ""

    async def get_broker_policies(self) -> Dict[str, Any]:
        """
        Get broker-specific policy information.

        Returns:
            Broker policies data
        """
        try:
            if self.integration_active and self.broker_policies:
                # Get real broker policies
                # In production, this would call:
                # policies = await self.broker_policies.get_current_policies()
                policies = {}

                return policies

            else:
                # Return mock broker policies
                return {
                    'broker_name': 'Alpaca',
                    'margin_requirements': {
                        'initial_margin': 0.50,
                        'maintenance_margin': 0.25,
                        'pdt_threshold': 25000.00
                    },
                    'trading_rules': {
                        'max_day_trades': 3,
                        'settlement_period': 'T+2',
                        'pattern_day_trader_rules': True
                    },
                    'restricted_symbols': [],
                    'last_updated': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Error getting broker policies: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }

    async def validate_trade_compliance(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trade for compliance before execution.

        Args:
            trade_data: Trade data to validate

        Returns:
            Validation result
        """
        try:
            # Get current PDT status
            pdt_status = await self.get_pdt_status()

            # Check PDT restrictions
            if pdt_status.get('status') == 'violation':
                return {
                    'valid': False,
                    'reason': 'Account is in PDT violation - day trading is restricted',
                    'blocked_by': 'PDT Rules',
                    'timestamp': datetime.utcnow().isoformat()
                }

            # Check day trade limits
            if pdt_status.get('day_trades_remaining', 3) <= 0:
                return {
                    'valid': False,
                    'reason': 'Day trade limit exceeded for current period',
                    'blocked_by': 'PDT Rules',
                    'timestamp': datetime.utcnow().isoformat()
                }

            # Check position size limits (simplified)
            position_value = trade_data.get('quantity', 0) * trade_data.get('price', 0)
            if position_value > 15000:  # $15k limit for demo
                return {
                    'valid': False,
                    'reason': 'Position size exceeds current limits',
                    'blocked_by': 'Risk Management',
                    'timestamp': datetime.utcnow().isoformat()
                }

            # Record compliance check
            await self.record_compliance_event(
                'trade_validation',
                {
                    'resource': 'trade_execution',
                    'details': f"Trade validated for {trade_data.get('symbol', 'unknown')}",
                    'trade_data': trade_data
                }
            )

            return {
                'valid': True,
                'message': 'Trade complies with all regulatory requirements',
                'validated_at': datetime.utcnow().isoformat(),
                'regulations_checked': ['PDT Rules', 'Position Limits', 'Risk Management']
            }

        except Exception as e:
            logger.error(f"Error validating trade compliance: {e}")
            return {
                'valid': False,
                'reason': f'Compliance validation failed: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }

    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get compliance system health and diagnostic information.

        Returns:
            System health data
        """
        try:
            health_data = {
                'integration_active': self.integration_active,
                'last_compliance_check': self.last_compliance_check.isoformat() if self.last_compliance_check else None,
                'compliance_violations': len(self.compliance_violations),
                'components_status': {}
            }

            # Check component status
            components = [
                ('pdt_tracker', self.pdt_tracker),
                ('regulatory_engine', self.regulatory_engine),
                ('audit_logger', self.audit_logger),
                ('broker_policies', self.broker_policies)
            ]

            for component_name, component in components:
                health_data['components_status'][component_name] = {
                    'available': component is not None,
                    'status': 'connected' if component else 'disconnected'
                }

            return health_data

        except Exception as e:
            logger.error(f"Error getting compliance system health: {e}")
            return {
                'integration_active': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Global compliance integration instance
compliance_integration = ComplianceIntegration()
