"""
Market Manipulation Detector

Detects potential market manipulation patterns including layering, spoofing, and wash trading.
"""

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from threading import Event
from typing import Dict, List, Optional, Any, Tuple
import json

from ..audit.audit_logger import AuditLogger


@dataclass
class ManipulationPattern:
    """Represents a detected manipulation pattern."""
    pattern_type: str
    confidence: float
    description: str
    indicators: List[str]
    timestamp: datetime
    symbol: str


class ManipulationDetector:
    """
    Market manipulation detection system.
    
    Detects various forms of market manipulation including:
    - Layering and spoofing
    - Wash trading
    - Pump and dump schemes
    - Front running
    - Quote stuffing
    """
    
    def __init__(self, shutdown_event: Optional[Event] = None):
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("manipulation_detector")
        self.audit_logger = AuditLogger()
        
        # Hard-coded manipulation detection parameters
        self._LAYERING_THRESHOLD = 5  # Orders per second
        self._SPOOFING_RATIO = Decimal('0.8')  # Cancelled vs executed ratio
        self._WASH_TRADE_MIN_PROFIT = Decimal('0.001')  # Minimum profit threshold
        self._QUOTE_STUFFING_THRESHOLD = 100  # Quotes per second
        self._PUMP_DUMP_VOLUME_SPIKE = 10  # Volume multiplier
        
        # Pattern tracking
        self._order_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._quote_patterns: Dict[str, List[datetime]] = {}
        self._volume_patterns: Dict[str, List[Tuple[datetime, int]]] = {}
        self._price_patterns: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        
        # Detection history
        self._detection_history: List[ManipulationPattern] = []
        
        self.logger.info("Manipulation detector initialized")
    
    async def initialize(self):
        """Initialize manipulation detector."""
        try:
            await self.audit_logger.initialize()
            self.logger.info("Manipulation detector initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize manipulation detector", error=str(e))
            raise
    
    async def shutdown(self):
        """Shutdown manipulation detector."""
        try:
            await self.audit_logger.shutdown()
            self.logger.info("Manipulation detector shutdown complete")
        except Exception as e:
            self.logger.error("Error during manipulation detector shutdown", error=str(e))
    
    async def detect(self, symbol: str, action: str, quantity: int,
                    price: Optional[Decimal] = None) -> Dict[str, Any]:
        """
        Detect potential market manipulation patterns.
        
        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Trade price
            
        Returns:
            Dict with 'detected' boolean and pattern details
        """
        if self.shutdown_event.is_set():
            return {'detected': False}
        
        patterns_detected = []
        confidence_scores = []
        
        try:
            # Record the order for pattern analysis
            await self._record_order_pattern(symbol, action, quantity, price)
            
            # Check for layering/spoofing
            layering_result = await self._detect_layering(symbol)
            if layering_result['detected']:
                patterns_detected.append(layering_result)
                confidence_scores.append(layering_result['confidence'])
            
            # Check for quote stuffing
            stuffing_result = await self._detect_quote_stuffing(symbol)
            if stuffing_result['detected']:
                patterns_detected.append(stuffing_result)
                confidence_scores.append(stuffing_result['confidence'])
            
            # Check for wash trading patterns
            wash_result = await self._detect_wash_trading_pattern(symbol, action, quantity, price)
            if wash_result['detected']:
                patterns_detected.append(wash_result)
                confidence_scores.append(wash_result['confidence'])
            
            # Check for pump and dump patterns
            pump_dump_result = await self._detect_pump_dump(symbol, price)
            if pump_dump_result['detected']:
                patterns_detected.append(pump_dump_result)
                confidence_scores.append(pump_dump_result['confidence'])
            
            # Check for front running
            front_running_result = await self._detect_front_running(symbol, action, price)
            if front_running_result['detected']:
                patterns_detected.append(front_running_result)
                confidence_scores.append(front_running_result['confidence'])
            
            if patterns_detected:
                # Calculate overall confidence
                overall_confidence = sum(confidence_scores) / len(confidence_scores)
                
                # Log detection for audit
                await self._log_manipulation_detection(symbol, patterns_detected, overall_confidence)
                
                return {
                    'detected': True,
                    'pattern': f"Multiple manipulation patterns: {', '.join([p['pattern_type'] for p in patterns_detected])}",
                    'confidence': overall_confidence,
                    'details': {
                        'patterns': patterns_detected,
                        'recommendation': 'BLOCK_TRADE' if overall_confidence > 0.8 else 'REVIEW_TRADE'
                    }
                }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error("Error during manipulation detection",
                            symbol=symbol,
                            action=action,
                            error=str(e))
            
            return {
                'detected': False,
                'error': str(e)
            }
    
    async def _record_order_pattern(self, symbol: str, action: str, quantity: int, price: Optional[Decimal]):
        """Record order pattern for analysis."""
        
        if symbol not in self._order_patterns:
            self._order_patterns[symbol] = []
        
        pattern = {
            'timestamp': datetime.now(timezone.utc),
            'action': action,
            'quantity': quantity,
            'price': float(price) if price else None
        }
        
        self._order_patterns[symbol].append(pattern)
        
        # Keep only recent patterns (last 5 minutes)
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        self._order_patterns[symbol] = [
            p for p in self._order_patterns[symbol] 
            if p['timestamp'] > cutoff_time
        ]
    
    async def _detect_layering(self, symbol: str) -> Dict[str, Any]:
        """Detect layering/spoofing patterns."""
        
        if symbol not in self._order_patterns:
            return {'detected': False}
        
        recent_orders = self._order_patterns[symbol]
        if len(recent_orders) < 10:
            return {'detected': False}
        
        # Check order frequency
        now = datetime.now(timezone.utc)
        recent_window = timedelta(seconds=1)
        orders_in_window = [
            o for o in recent_orders 
            if (now - o['timestamp']) < recent_window
        ]
        
        if len(orders_in_window) >= self._LAYERING_THRESHOLD:
            return {
                'detected': True,
                'pattern_type': 'LAYERING',
                'confidence': 0.8,
                'description': f'High frequency orders detected: {len(orders_in_window)} orders in 1 second',
                'indicators': ['high_frequency', 'potential_spoofing']
            }
        
        return {'detected': False}
    
    async def _detect_quote_stuffing(self, symbol: str) -> Dict[str, Any]:
        """Detect quote stuffing patterns."""
        
        if symbol not in self._quote_patterns:
            return {'detected': False}
        
        # This would normally track quote updates from market data feed
        # For now, we'll use order patterns as a proxy
        quote_frequency = len(self._quote_patterns[symbol])
        
        if quote_frequency >= self._QUOTE_STUFFING_THRESHOLD:
            return {
                'detected': True,
                'pattern_type': 'QUOTE_STUFFING',
                'confidence': 0.7,
                'description': f'Excessive quote updates: {quote_frequency} per second',
                'indicators': ['quote_stuffing', 'market_disruption']
            }
        
        return {'detected': False}
    
    async def _detect_wash_trading_pattern(self, symbol: str, action: str, quantity: int,
                                         price: Optional[Decimal]) -> Dict[str, Any]:
        """Detect wash trading patterns."""
        
        if symbol not in self._order_patterns:
            return {'detected': False}
        
        recent_orders = self._order_patterns[symbol]
        
        # Look for rapid buy/sell patterns in same symbol
        buy_orders = [o for o in recent_orders if o['action'] == 'buy'][-10:]
        sell_orders = [o for o in recent_orders if o['action'] == 'sell'][-10:]
        
        if not buy_orders or not sell_orders:
            return {'detected': False}
        
        # Check for alternating buy/sell patterns
        alternating_pattern = 0
        for i in range(min(len(buy_orders), len(sell_orders)) - 1):
            if ((buy_orders[i]['timestamp'] < sell_orders[i]['timestamp'] < buy_orders[i+1]['timestamp']) or
                (sell_orders[i]['timestamp'] < buy_orders[i]['timestamp'] < sell_orders[i+1]['timestamp'])):
                alternating_pattern += 1
        
        if alternating_pattern >= 3:
            return {
                'detected': True,
                'pattern_type': 'WASH_TRADING',
                'confidence': 0.9,
                'description': f'Wash trading pattern detected: {alternating_pattern} alternating trades',
                'indicators': ['alternating_pattern', 'artificial_volume']
            }
        
        return {'detected': False}
    
    async def _detect_pump_dump(self, symbol: str, price: Optional[Decimal]) -> Dict[str, Any]:
        """Detect pump and dump patterns."""
        
        if not price or symbol not in self._price_patterns:
            return {'detected': False}
        
        price_history = self._price_patterns[symbol][-20:]  # Last 20 price points
        
        if len(price_history) < 10:
            return {'detected': False}
        
        # Calculate price volatility
        prices = [p[1] for p in price_history]
        avg_price = sum(prices) / len(prices)
        
        if avg_price == 0:
            return {'detected': False}
        
        # Check for unusual price spike
        price_changes = []
        for i in range(1, len(prices)):
            change = (prices[i] - prices[i-1]) / prices[i-1]
            price_changes.append(abs(change))
        
        avg_change = sum(price_changes) / len(price_changes)
        max_change = max(price_changes)
        
        if max_change > avg_change * 5:  # 5x normal volatility
            return {
                'detected': True,
                'pattern_type': 'PUMP_DUMP',
                'confidence': 0.6,
                'description': f'Unusual price volatility: {max_change:.2%} vs average {avg_change:.2%}',
                'indicators': ['price_spike', 'high_volatility']
            }
        
        return {'detected': False}
    
    async def _detect_front_running(self, symbol: str, action: str, price: Optional[Decimal]) -> Dict[str, Any]:
        """Detect front running patterns."""
        
        # This would normally require order flow analysis
        # For now, we'll use simplified heuristics
        
        if symbol not in self._order_patterns:
            return {'detected': False}
        
        recent_orders = self._order_patterns[symbol][-5:]
        
        # Look for patterns where large orders precede price movements
        if len(recent_orders) >= 3:
            # Check if we have a pattern of increasing order sizes
            quantities = [o['quantity'] for o in recent_orders]
            if quantities == sorted(quantities) and quantities[-1] > quantities[0] * 2:
                return {
                    'detected': True,
                    'pattern_type': 'FRONT_RUNNING',
                    'confidence': 0.5,
                    'description': 'Potential front running: increasing order sizes before price movement',
                    'indicators': ['order_size_escalation', 'timing_pattern']
                }
        
        return {'detected': False}
    
    async def _log_manipulation_detection(self, symbol: str, patterns: List[Dict[str, Any]], confidence: float):
        """Log manipulation detection for audit trail."""
        
        detection_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'manipulation_detection',
            'symbol': symbol,
            'overall_confidence': confidence,
            'patterns_detected': len(patterns),
            'pattern_details': patterns
        }
        
        await self.audit_logger.log_compliance_violation(detection_data)
        
        # Store in history
        for pattern in patterns:
            manipulation_pattern = ManipulationPattern(
                pattern_type=pattern['pattern_type'],
                confidence=pattern['confidence'],
                description=pattern['description'],
                indicators=pattern['indicators'],
                timestamp=datetime.now(timezone.utc),
                symbol=symbol
            )
            self._detection_history.append(manipulation_pattern)
            
            # Keep only recent history
            if len(self._detection_history) > 1000:
                self._detection_history = self._detection_history[-500:]
    
    def get_detection_history(self) -> List[Dict[str, Any]]:
        """Get recent manipulation detection history."""
        
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
        """Get manipulation detection statistics."""
        
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
