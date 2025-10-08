"""
Session Manager for PDT Trading Bot Admin UI

This module provides session management, rate limiting, and user state
tracking for secure authentication and authorization.
"""

import time
import hashlib
from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

@dataclass
class UserSession:
    """Represents a user session with metadata."""
    user_id: str
    username: str
    role: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    login_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class RateLimitEntry:
    """Represents a rate limiting entry."""
    attempts: int = 0
    window_start: float = field(default_factory=time.time)
    blocked_until: Optional[float] = None

class SessionManager:
    """Manages user sessions, authentication state, and rate limiting."""

    def __init__(self):
        """Initialize session manager with configuration."""
        self.sessions: Dict[str, UserSession] = {}
        self.token_blacklist: Set[str] = set()
        self.login_attempts: Dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self.failed_logins: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

        # Rate limiting configuration
        self.max_login_attempts = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
        self.lockout_duration = int(os.getenv("LOCKOUT_DURATION_MINUTES", "15"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "300"))  # 5 minutes
        self.max_requests_per_window = int(os.getenv("MAX_REQUESTS_PER_WINDOW", "100"))

        # Session timeout configuration
        self.session_timeout = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
        self.cleanup_interval = int(os.getenv("SESSION_CLEANUP_INTERVAL_MINUTES", "10"))

        # Threading lock for thread safety
        self._lock = threading.RLock()

        # Start cleanup thread
        self._start_cleanup_thread()

    def create_session(self, user_id: str, username: str, role: str,
                      ip_address: str, user_agent: str) -> str:
        """
        Create a new user session.

        Args:
            user_id: Unique user identifier
            username: Username
            role: User role
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Session ID
        """
        with self._lock:
            session_id = self._generate_session_id(user_id, ip_address)

            session = UserSession(
                user_id=user_id,
                username=username,
                role=role,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent
            )

            self.sessions[session_id] = session
            return session_id

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            UserSession if found and active, None otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)

            if not session or not session.is_active:
                return None

            # Check if session has expired
            if self._is_session_expired(session):
                self.invalidate_session(session_id)
                return None

            # Update last activity
            session.last_activity = datetime.utcnow()
            return session

    def update_session_activity(self, session_id: str) -> bool:
        """
        Update the last activity time for a session.

        Args:
            session_id: Session ID

        Returns:
            True if session exists and was updated, False otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                session.last_activity = datetime.utcnow()
                return True
            return False

    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session.

        Args:
            session_id: Session ID to invalidate

        Returns:
            True if session was invalidated, False if not found
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.is_active = False
                return True
            return False

    def invalidate_user_sessions(self, user_id: str) -> int:
        """
        Invalidate all sessions for a specific user.

        Args:
            user_id: User ID

        Returns:
            Number of sessions invalidated
        """
        with self._lock:
            invalidated_count = 0
            sessions_to_remove = []

            for session_id, session in self.sessions.items():
                if session.user_id == user_id and session.is_active:
                    session.is_active = False
                    sessions_to_remove.append(session_id)
                    invalidated_count += 1

            # Remove invalidated sessions
            for session_id in sessions_to_remove:
                del self.sessions[session_id]

            return invalidated_count

    def check_rate_limit(self, identifier: str, max_attempts: int = None,
                        window_seconds: int = None) -> Tuple[bool, Optional[float]]:
        """
        Check if an action is within rate limits.

        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            max_attempts: Maximum attempts allowed (optional)
            window_seconds: Time window in seconds (optional)

        Returns:
            Tuple of (is_allowed, seconds_until_reset)
        """
        if max_attempts is None:
            max_attempts = self.max_login_attempts
        if window_seconds is None:
            window_seconds = self.rate_limit_window

        with self._lock:
            current_time = time.time()
            entry = self.login_attempts[identifier]

            # Reset window if expired
            if current_time - entry.window_start > window_seconds:
                entry.attempts = 0
                entry.window_start = current_time

            # Check if currently blocked
            if entry.blocked_until and current_time < entry.blocked_until:
                return False, entry.blocked_until - current_time

            # Check attempt limit
            if entry.attempts >= max_attempts:
                entry.blocked_until = current_time + (self.lockout_duration * 60)
                return False, self.lockout_duration * 60

            return True, None

    def record_failed_attempt(self, identifier: str) -> None:
        """
        Record a failed login attempt for rate limiting.

        Args:
            identifier: Unique identifier (IP, username, etc.)
        """
        with self._lock:
            entry = self.login_attempts[identifier]
            entry.attempts += 1

            # Record in failed logins history
            self.failed_logins[identifier].append(time.time())

    def record_successful_login(self, identifier: str) -> None:
        """
        Record a successful login (resets rate limiting).

        Args:
            identifier: Unique identifier (IP, username, etc.)
        """
        with self._lock:
            if identifier in self.login_attempts:
                self.login_attempts[identifier].attempts = 0
                self.login_attempts[identifier].blocked_until = None

    def is_account_locked(self, identifier: str) -> Tuple[bool, Optional[float]]:
        """
        Check if an account is currently locked due to failed attempts.

        Args:
            identifier: Unique identifier (IP, username, etc.)

        Returns:
            Tuple of (is_locked, seconds_until_unlock)
        """
        with self._lock:
            entry = self.login_attempts.get(identifier)
            if not entry or not entry.blocked_until:
                return False, None

            current_time = time.time()
            if current_time < entry.blocked_until:
                return True, entry.blocked_until - current_time

            return False, None

    def get_active_sessions(self, user_id: Optional[str] = None) -> Dict[str, UserSession]:
        """
        Get all active sessions, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Dictionary of active sessions
        """
        with self._lock:
            active_sessions = {}

            for session_id, session in self.sessions.items():
                if session.is_active and not self._is_session_expired(session):
                    if user_id is None or session.user_id == user_id:
                        active_sessions[session_id] = session

            return active_sessions

    def get_session_count(self, user_id: Optional[str] = None) -> int:
        """
        Get the count of active sessions.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Number of active sessions
        """
        return len(self.get_active_sessions(user_id))

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired and inactive sessions.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            expired_sessions = []
            current_time = datetime.utcnow()

            for session_id, session in self.sessions.items():
                # Check for expired sessions
                if (not session.is_active or
                    current_time - session.last_activity >
                    timedelta(minutes=self.session_timeout)):
                    expired_sessions.append(session_id)

            # Remove expired sessions
            for session_id in expired_sessions:
                del self.sessions[session_id]

            return len(expired_sessions)

    def _generate_session_id(self, user_id: str, ip_address: str) -> str:
        """Generate a unique session ID."""
        timestamp = str(time.time())
        data = f"{user_id}:{ip_address}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def _is_session_expired(self, session: UserSession) -> bool:
        """Check if a session has expired."""
        return (datetime.utcnow() - session.last_activity >
                timedelta(minutes=self.session_timeout))

    def _start_cleanup_thread(self) -> None:
        """Start background thread for session cleanup."""
        def cleanup_task():
            while True:
                try:
                    time.sleep(self.cleanup_interval * 60)
                    self.cleanup_expired_sessions()
                except Exception as e:
                    print(f"Error in session cleanup: {e}")

        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

    def get_security_metrics(self) -> Dict:
        """
        Get security metrics for monitoring.

        Returns:
            Dictionary containing security metrics
        """
        with self._lock:
            active_sessions = len(self.get_active_sessions())
            locked_accounts = sum(1 for entry in self.login_attempts.values()
                                if entry.blocked_until and time.time() < entry.blocked_until)

            return {
                "active_sessions": active_sessions,
                "total_sessions": len(self.sessions),
                "locked_accounts": locked_accounts,
                "blacklisted_tokens": len(self.token_blacklist),
                "failed_login_attempts": sum(entry.attempts for entry in self.login_attempts.values())
            }

    def blacklist_token(self, token_jti: str) -> None:
        """
        Add a token to the blacklist.

        Args:
            token_jti: Token's unique identifier
        """
        with self._lock:
            self.token_blacklist.add(token_jti)

    def is_token_blacklisted(self, token_jti: str) -> bool:
        """
        Check if a token is blacklisted.

        Args:
            token_jti: Token's unique identifier

        Returns:
            True if blacklisted, False otherwise
        """
        with self._lock:
            return token_jti in self.token_blacklist

    def clear_blacklisted_tokens(self) -> int:
        """
        Clear all blacklisted tokens.

        Returns:
            Number of tokens cleared
        """
        with self._lock:
            count = len(self.token_blacklist)
            self.token_blacklist.clear()
            return count
