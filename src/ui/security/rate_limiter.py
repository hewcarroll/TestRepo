"""
Rate Limiter Module for PDT Trading Bot Admin UI

This module provides rate limiting functionality to protect against
brute force attacks, abuse, and excessive API usage.
"""

import time
import hashlib
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimitEntry:
    """Rate limit entry for tracking requests."""
    requests: deque = field(default_factory=lambda: deque(maxlen=100))
    blocked_until: Optional[float] = None
    total_requests: int = 0

@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    requests_per_window: int
    window_seconds: int
    block_duration_seconds: int
    burst_limit: int = 10

class RateLimiter:
    """Advanced rate limiter with multiple strategies."""

    def __init__(self):
        """Initialize rate limiter."""
        self.entries: Dict[str, RateLimitEntry] = {}
        self.ip_whitelist: set = set()
        self.ip_blacklist: set = set()
        self.user_limits: Dict[str, RateLimitRule] = {}
        self.global_rules: Dict[str, RateLimitRule] = {}

        # Threading lock for thread safety
        self._lock = threading.RLock()

        # Default global rules
        self._setup_default_rules()

        # Cleanup thread
        self._start_cleanup_thread()

    def _setup_default_rules(self) -> None:
        """Set up default rate limiting rules."""
        # Global API limits
        self.global_rules['api_general'] = RateLimitRule(
            requests_per_window=1000,
            window_seconds=300,  # 5 minutes
            block_duration_seconds=900,  # 15 minutes
            burst_limit=50
        )

        # Authentication limits (stricter)
        self.global_rules['auth'] = RateLimitRule(
            requests_per_window=5,
            window_seconds=300,  # 5 minutes
            block_duration_seconds=1800,  # 30 minutes
            burst_limit=3
        )

        # Trading control limits
        self.global_rules['trading_control'] = RateLimitRule(
            requests_per_window=10,
            window_seconds=60,  # 1 minute
            block_duration_seconds=300,  # 5 minutes
            burst_limit=3
        )

        # Emergency controls (very strict)
        self.global_rules['emergency'] = RateLimitRule(
            requests_per_window=3,
            window_seconds=60,  # 1 minute
            block_duration_seconds=600,  # 10 minutes
            burst_limit=1
        )

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_task():
            while True:
                try:
                    time.sleep(300)  # Clean up every 5 minutes
                    self._cleanup_old_entries()
                except Exception as e:
                    logger.error(f"Error in rate limiter cleanup: {e}")

        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

    def _cleanup_old_entries(self) -> None:
        """Clean up old rate limit entries."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - 3600  # Remove entries older than 1 hour

            keys_to_remove = []
            for key, entry in self.entries.items():
                # Remove old requests from deque
                while entry.requests and entry.requests[0] < cutoff_time:
                    entry.requests.popleft()

                # If no recent requests, mark for removal
                if not entry.requests:
                    keys_to_remove.append(key)

            # Remove empty entries
            for key in keys_to_remove:
                del self.entries[key]

    def _get_identifier_key(self, identifier: str, rule_type: str) -> str:
        """Generate unique key for rate limiting."""
        return hashlib.sha256(f"{identifier}:{rule_type}".encode()).hexdigest()[:16]

    def check_rate_limit(self, identifier: str, rule_type: str = 'api_general',
                        custom_rule: Optional[RateLimitRule] = None) -> Tuple[bool, Optional[float]]:
        """
        Check if request is within rate limits.

        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            rule_type: Type of rate limit rule to apply
            custom_rule: Custom rule to override defaults

        Returns:
            Tuple of (is_allowed, seconds_until_reset)
        """
        with self._lock:
            # Check blacklists first
            if identifier in self.ip_blacklist:
                return False, 3600  # Block for 1 hour

            # Check whitelists
            if identifier in self.ip_whitelist:
                return True, None

            # Get rule
            rule = custom_rule or self.global_rules.get(rule_type)
            if not rule:
                rule = self.global_rules['api_general']

            # Get or create entry
            key = self._get_identifier_key(identifier, rule_type)
            if key not in self.entries:
                self.entries[key] = RateLimitEntry()

            entry = self.entries[key]
            current_time = time.time()

            # Check if currently blocked
            if entry.blocked_until and current_time < entry.blocked_until:
                return False, entry.blocked_until - current_time

            # Clean old requests
            window_start = current_time - rule.window_seconds
            while entry.requests and entry.requests[0] < window_start:
                entry.requests.popleft()

            # Check burst limit (requests in last 10 seconds)
            recent_requests = [req for req in entry.requests if req > current_time - 10]
            if len(recent_requests) >= rule.burst_limit:
                entry.blocked_until = current_time + rule.block_duration_seconds
                logger.warning(f"Burst limit exceeded for {identifier}")
                return False, rule.block_duration_seconds

            # Check window limit
            if len(entry.requests) >= rule.requests_per_window:
                entry.blocked_until = current_time + rule.block_duration_seconds
                logger.warning(f"Rate limit exceeded for {identifier}")
                return False, rule.block_duration_seconds

            # Record this request
            entry.requests.append(current_time)
            entry.total_requests += 1

            return True, None

    def add_to_whitelist(self, identifier: str) -> None:
        """Add identifier to whitelist."""
        with self._lock:
            self.ip_whitelist.add(identifier)
            logger.info(f"Added {identifier} to rate limit whitelist")

    def remove_from_whitelist(self, identifier: str) -> None:
        """Remove identifier from whitelist."""
        with self._lock:
            self.ip_whitelist.discard(identifier)
            logger.info(f"Removed {identifier} from rate limit whitelist")

    def add_to_blacklist(self, identifier: str, duration: int = 3600) -> None:
        """Add identifier to blacklist."""
        with self._lock:
            self.ip_blacklist.add(identifier)
            logger.warning(f"Added {identifier} to rate limit blacklist for {duration} seconds")

            # Auto-remove from blacklist after duration
            def remove_from_blacklist():
                time.sleep(duration)
                with self._lock:
                    self.ip_blacklist.discard(identifier)
                    logger.info(f"Removed {identifier} from rate limit blacklist")

            threading.Thread(target=remove_from_blacklist, daemon=True).start()

    def set_user_limit(self, user_id: str, rule: RateLimitRule) -> None:
        """Set custom rate limit for specific user."""
        with self._lock:
            self.user_limits[user_id] = rule
            logger.info(f"Set custom rate limit for user {user_id}")

    def get_rate_limit_status(self, identifier: str, rule_type: str = 'api_general') -> Dict[str, Any]:
        """
        Get rate limit status for identifier.

        Args:
            identifier: Identifier to check
            rule_type: Rule type to check

        Returns:
            Rate limit status information
        """
        with self._lock:
            key = self._get_identifier_key(identifier, rule_type)
            entry = self.entries.get(key)

            if not entry:
                return {
                    'requests_in_window': 0,
                    'window_remaining': 0,
                    'is_blocked': False,
                    'blocked_until': None,
                    'total_requests': 0
                }

            current_time = time.time()
            rule = self.global_rules.get(rule_type, self.global_rules['api_general'])

            # Count requests in current window
            window_start = current_time - rule.window_seconds
            requests_in_window = len([req for req in entry.requests if req > window_start])

            return {
                'requests_in_window': requests_in_window,
                'window_remaining': rule.window_seconds - (current_time - (entry.requests[0] if entry.requests else current_time)),
                'is_blocked': entry.blocked_until and current_time < entry.blocked_until,
                'blocked_until': entry.blocked_until,
                'total_requests': entry.total_requests,
                'limit': rule.requests_per_window,
                'window_seconds': rule.window_seconds
            }

    def reset_rate_limit(self, identifier: str, rule_type: str = 'api_general') -> None:
        """Reset rate limit for identifier."""
        with self._lock:
            key = self._get_identifier_key(identifier, rule_type)
            if key in self.entries:
                del self.entries[key]
                logger.info(f"Reset rate limit for {identifier}")

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        with self._lock:
            total_requests = sum(entry.total_requests for entry in self.entries.values())
            blocked_count = sum(1 for entry in self.entries.values() if entry.blocked_until and time.time() < entry.blocked_until)
            active_entries = len(self.entries)

            return {
                'total_requests': total_requests,
                'blocked_identifiers': blocked_count,
                'active_entries': active_entries,
                'whitelist_size': len(self.ip_whitelist),
                'blacklist_size': len(self.ip_blacklist),
                'timestamp': datetime.utcnow().isoformat()
            }

# Global rate limiter instance
rate_limiter = RateLimiter()

# Convenience functions for common use cases
def check_api_rate_limit(identifier: str) -> Tuple[bool, Optional[float]]:
    """Check general API rate limit."""
    return rate_limiter.check_rate_limit(identifier, 'api_general')

def check_auth_rate_limit(identifier: str) -> Tuple[bool, Optional[float]]:
    """Check authentication rate limit."""
    return rate_limiter.check_rate_limit(identifier, 'auth')

def check_trading_rate_limit(identifier: str) -> Tuple[bool, Optional[float]]:
    """Check trading control rate limit."""
    return rate_limiter.check_rate_limit(identifier, 'trading_control')

def check_emergency_rate_limit(identifier: str) -> Tuple[bool, Optional[float]]:
    """Check emergency controls rate limit."""
    return rate_limiter.check_rate_limit(identifier, 'emergency')
