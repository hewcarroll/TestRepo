"""
Wash Trade Detector

Detects wash trading patterns and self-trading activities that violate market integrity rules.
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from threading import Event
from typing import Dict, List, Optional, Any, Set, Tuple
import json

from ..audit.audit_logger import AuditLogger


@dataclass
class WashTradePattern:
    """Represents a detected wash trade pattern."""
    pattern_type: str
    confidence: float
    description: str
    indicators: List[str]
    timestamp: datetime
    symbol: str
    accounts_involved: List[str]
    trade_sequence: List[Dict[str, Any]]


class WashTradeDetector:
    """
    Wash trading detection system.
    
    Detects various forms of wash trading including:
    - Self-trading between related accounts
    - Coordinated trading patterns
    - Artificial volume creation
    - Cross trading violations
    """
    
    def __init__(self, shutdown_event: Optional[Event] = None):
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("wash_trade_detector")
        self.audit_logger = AuditLogger()
        
        # Hard-coded wash trade detection parameters
        self._WASH_TRADE_TIME_WINDOW = 60  # seconds
        self._MIN_PROFIT_THRESHOLD = Decimal('0.001')  # Minimum profit to be considered wash
        self._SAME_ACCOUNT_THRESHOLD = 5  # Same account trades in window
        self._COORDINATED_TRADE_RATIO = Decimal('0.9')  # Ratio of coordinated vs total trades
        self._ARTIFICIAL_VOLUME_THRESHOLD = 1000  # Minimum volume for artificial volume detection
        
        # Trade tracking for wash detection
        self._account_trades: Dict[str, List[Dict[str, Any]]] = {}
        self._symbol_trades: Dict[str, List[Dict[str, Any]]] = {}
        self._cross_trades: Dict[str, List[Tuple[str, str, datetime]]] = {}  # symbol -> (buy_account, sell_account, time)
        
        # Account relationship mapping (would be populated from account database)
        self._related_accounts: Dict[str, Set[str]] = {}
        
        # Detection history
        self._detection_history: List[WashTradePattern] = []
        
        self.logger.info("Wash trade detector initialized")
    
    async def initialize(self):
        """Initialize wash trade detector."""
        try:
            await self.audit_logger.initialize()
            await self._load_account_relationships()
            self.logger.info("Wash trade detector initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize wash trade detector", error=str(e))
            raise
    
    async def shutdown(self):
        """Shutdown wash trade detector."""
        try:
            await self.audit_logger.shutdown()
            self.logger.info("Wash trade detector shutdown complete")
        except Exception as e:
            self.logger.error("Error during wash trade detector shutdown", error=str(e))
    
    async def detect(self, symbol: str, action: str, quantity: int,
                    current_positions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect potential wash trading patterns.
        
        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            current_positions: Current portfolio positions
            
        Returns:
            Dict with 'detected' boolean and pattern details
        """
        if self.shutdown_event.is_set():
            return {'detected': False}
        
        patterns_detected = []
        confidence_scores = []
        
        try:
            # Record the trade for pattern analysis
            await self._record_trade(symbol, action, quantity)
            
            # Check for self-trading patterns
            self_trade_result = await self._detect_self_trading(symbol, action, quantity)
            if self_trade_result['detected']:
                patterns_detected.append(self_trade_result)
                confidence_scores.append(self_trade_result['confidence'])
            
            # Check for coordinated trading
            coordinated_result = await self._detect_coordinated_trading(symbol)
            if coordinated_result['detected']:
                patterns_detected.append(coordinated_result)
                confidence_scores.append(coordinated_result['confidence'])
            
            # Check for artificial volume patterns
            volume_result = await self._detect_artificial_volume(symbol)
            if volume_result['detected']:
                patterns_detected.append(volume_result)
                confidence_scores.append(volume_result['confidence'])
            
            # Check for cross trading violations
            cross_trade_result = await self._detect_cross_trading(symbol, action, quantity)
            if cross_trade_result['detected']:
                patterns_detected.append(cross_trade_result)
                confidence_scores.append(cross_trade_result['confidence'])
            
            if patterns_detected:
                # Calculate overall confidence
                overall_confidence = sum(confidence_scores) / len(confidence_scores)
                
                # Log detection for audit
                await self._log_wash_trade_detection(symbol, patterns_detected, overall_confidence)
                
                return {
                    'detected': True,
                    'reason': f"Wash trading patterns detected: {', '.join([p['pattern_type'] for p in patterns_detected])}",
                    'confidence': overall_confidence,
                    'details': {
                        'patterns': patterns_detected,
                        'recommendation': 'BLOCK_TRADE' if overall_confidence > 0.8 else 'REVIEW_TRADE'
                    }
                }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error("Error during wash trade detection",
                            symbol=symbol,
                            action=action,
                            error=str(e))
            
            return {
                'detected': False,
                'error': str(e)
            }
    
    async def _record_trade(self, symbol: str, action: str, quantity: int):
        """Record trade for wash trading analysis."""
        
        # In a real implementation, this would include account ID
        # For now, we'll use a placeholder account ID
        account_id = "PRIMARY_ACCOUNT"  # This would come from the actual trading context
        
        trade_record = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'account_id': account_id
        }
        
        # Record by account
        if account_id not in self._account_trades:
            self._account_trades[account_id] = []
        
        self._account_trades[account_id].append(trade_record)
        
        # Record by symbol
        if symbol not in self._symbol_trades:
            self._symbol_trades[symbol] = []
        
        self._symbol_trades[symbol].append(trade_record)
        
        # Clean old records (keep last 5 minutes)
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self._WASH_TRADE_TIME_WINDOW)
        
        self._account_trades[account_id] = [
            t for t in self._account_trades[account_id] 
            if t['timestamp'] > cutoff_time
        ]
        
        self._symbol_trades[symbol] = [
            t for t in self._symbol_trades[symbol] 
            if t['timestamp'] > cutoff_time
        ]
    
    async def _detect_self_trading(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Detect self-trading patterns within the same account."""
        
        account_id = "PRIMARY_ACCOUNT"  # This would come from actual trading context
        
        if account_id not in self._account_trades:
            return {'detected': False}
        
        recent_trades = self._account_trades[account_id]
        
        # Look for rapid buy/sell in same symbol by same account
        symbol_trades = [t for t in recent_trades if t['symbol'] == symbol]
        
        if len(symbol_trades) < 2:
            return {'detected': False}
        
        # Check for alternating buy/sell pattern
        actions = [t['action'] for t in symbol_trades[-10:]]
        alternating_count = 0
        
        for i in range(len(actions) - 1):
            if actions[i] != actions[i + 1]:
                alternating_count += 1
        
        if alternating_count >= 3:
            return {
                'detected': True,
                'pattern_type': 'SELF_TRADING',
                'confidence': 0.9,
                'description': f'Self-trading pattern detected: {alternating_count} alternating trades',
                'indicators': ['alternating_pattern', 'same_account']
            }
        
        return {'detected': False}
    
    async def _detect_coordinated_trading(self, symbol: str) -> Dict[str, Any]:
        """Detect coordinated trading between related accounts."""
        
        if symbol not in self._symbol_trades:
            return {'detected': False}
        
        recent_trades = self._symbol_trades[symbol]
        
        if len(recent_trades) < 10:
            return {'detected': False}
        
        # Check for highly coordinated patterns
        # Look for accounts that consistently trade opposite sides
        account_actions = {}
        for trade in recent_trades[-20:]:
            account_id = trade['account_id']
            if account_id not in account_actions:
                account_actions[account_id] = []
            account_actions[account_id].append(trade['action'])
        
        # Check for accounts with opposite trading patterns
        coordinated_accounts = []
        for account1, actions1 in account_actions.items():
            for account2, actions2 in account_actions.items():
                if account1 != account2:
                    # Check if accounts consistently take opposite sides
                    opposite_trades = 0
                    total_compared = 0
                    
                    for a1, a2 in zip(actions1[-10:], actions2[-10:]):
                        if a1 != a2:
                            opposite_trades += 1
                        total_compared += 1
                    
                    if total_compared > 0:
                        opposite_ratio = opposite_trades / total_compared
                        if opposite_ratio >= self._COORDINATED_TRADE_RATIO:
                            coordinated_accounts.extend([account1, account2])
        
        if coordinated_accounts:
            return {
                'detected': True,
                'pattern_type': 'COORDINATED_TRADING',
                'confidence': 0.7,
                'description': f'Coordinated trading detected between accounts: {set(coordinated_accounts)}',
                'indicators': ['opposite_patterns', 'related_accounts']
            }
        
        return {'detected': False}
    
    async def _detect_artificial_volume(self, symbol: str) -> Dict[str, Any]:
        """Detect artificial volume creation patterns."""
        
        if symbol not in self._symbol_trades:
            return {'detected': False}
        
        recent_trades = self._symbol_trades[symbol]
        
        if len(recent_trades) < 20:
            return {'detected': False}
        
        # Calculate volume metrics
        total_volume = sum(t['quantity'] for t in recent_trades)
        unique_accounts = len(set(t['account_id'] for t in recent_trades))
        
        # Check for high volume with few accounts (potential artificial volume)
        if total_volume > self._ARTIFICIAL_VOLUME_THRESHOLD and unique_accounts < 3:
            # Check for repetitive patterns
            avg_quantity = total_volume / len(recent_trades)
            quantity_variance = sum((t['quantity'] - avg_quantity) ** 2 for t in recent_trades) / len(recent_trades)
            
            if quantity_variance < avg_quantity * 0.1:  # Low variance suggests artificial pattern
                return {
                    'detected': True,
                    'pattern_type': 'ARTIFICIAL_VOLUME',
                    'confidence': 0.8,
                    'description': f'Artificial volume pattern: {total_volume} volume across {unique_accounts} accounts',
                    'indicators': ['high_volume_low_accounts', 'repetitive_quantities']
                }
        
        return {'detected': False}
    
    async def _detect_cross_trading(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        """Detect cross trading violations."""
        
        # In a real implementation, this would check for trades between 
        # accounts with common ownership or control
        
        account_id = "PRIMARY_ACCOUNT"  # This would come from actual trading context
        
        # Check if this account has related accounts trading the opposite side
        if account_id in self._related_accounts:
            related_accounts = self._related_accounts[account_id]
            
            # Look for opposite trades in related accounts
            if symbol in self._symbol_trades:
                recent_trades = self._symbol_trades[symbol]
                cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self._WASH_TRADE_TIME_WINDOW)
                
                account_trade = None
                related_trade = None
                
                for trade in recent_trades:
                    if trade['timestamp'] > cutoff_time:
                        if trade['account_id'] == account_id:
                            account_trade = trade
                        elif trade['account_id'] in related_accounts:
                            related_trade = trade
                
                if account_trade and related_trade:
                    if account_trade['action'] != related_trade['action']:
                        return {
                            'detected': True,
                            'pattern_type': 'CROSS_TRADING',
                            'confidence': 0.9,
                            'description': f'Cross trading between related accounts: {account_id} and {related_trade["account_id"]}',
                            'indicators': ['related_accounts', 'opposite_sides']
                        }
        
        return {'detected': False}
    
    async def _log_wash_trade_detection(self, symbol: str, patterns: List[Dict[str, Any]], confidence: float):
        """Log wash trade detection for audit trail."""
        
        detection_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'wash_trade_detection',
            'symbol': symbol,
            'overall_confidence': confidence,
            'patterns_detected': len(patterns),
            'pattern_details': patterns
        }
        
        await self.audit_logger.log_compliance_violation(detection_data)
        
        # Store in history
        for pattern in patterns:
            wash_pattern = WashTradePattern(
                pattern_type=pattern['pattern_type'],
                confidence=pattern['confidence'],
                description=pattern['description'],
                indicators=pattern['indicators'],
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                accounts_involved=[],  # Would be populated in real implementation
                trade_sequence=[]  # Would be populated in real implementation
            )
            self._detection_history.append(wash_pattern)
            
            # Keep only recent history
            if len(self._detection_history) > 1000:
                self._detection_history = self._detection_history[-500:]
    
    async def _load_account_relationships(self):
        """Load account relationship data for cross trading detection."""
        try:
            # In a real implementation, this would load from account database
            # For now, we'll initialize with empty relationships
            self.logger.info("Account relationships loaded")
        except Exception as e:
            self.logger.error("Error loading account relationships", error=str(e))
    
    def get_detection_history(self) -> List[Dict[str, Any]]:
        """Get recent wash trade detection history."""
        
        return [
            {
                'pattern_type': p.pattern_type,
                'confidence': p.confidence,
                'description': p.description,
                'timestamp': p.timestamp.isoformat(),
                'symbol': p.symbol
            }
            for p in self._detection_history[-100:]  # Last 100 detections
        ]
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get wash trade detection statistics."""
        
        if not self._detection_history:
            return {'total_detections': 0}
        
        # Group by pattern type
        pattern_counts = {}
        for pattern in self._detection_history:
            pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
        
        return {
            'total_detections': len(self._detection_history),
            'pattern_breakdown': pattern_counts,
            'average_confidence': sum(p.confidence for p in self._detection_history) / len(self._detection_history),
            'recent_detections': len([p for p in self._detection_history if (datetime.now(timezone.utc) - p.timestamp).total_seconds() < 3600])
        }
